use std::collections::BTreeMap;
use std::iter::Peekable;
use std::str::Chars;

use anyhow::{anyhow, bail, Result};
use smallvec::SmallVec;

use crate::index::fastfields::FastFieldsReader;
use crate::index::manifest::Schema;
use crate::query::util::ensure_numeric_fast;
use crate::DocId;

const MAX_SCRIPT_LENGTH: usize = 512;
const MAX_SCRIPT_TOKENS: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
  Add,
  Sub,
  Mul,
  Div,
  Neg,
}

impl Op {
  fn precedence(self) -> u8 {
    match self {
      Op::Add | Op::Sub => 1,
      Op::Mul | Op::Div => 2,
      Op::Neg => 3,
    }
  }

  fn is_right_associative(self) -> bool {
    matches!(self, Op::Neg)
  }
}

#[derive(Clone, Debug)]
enum Token {
  Number(f64),
  Ident(String),
  LParen,
  RParen,
  Op(Op),
}

#[derive(Clone, Debug)]
enum Instruction {
  PushConst(f64),
  PushParam(usize),
  PushField(usize),
  PushScore,
  Add,
  Sub,
  Mul,
  Div,
  Neg,
}

#[derive(Clone, Debug)]
pub(crate) struct CompiledScript {
  instructions: Vec<Instruction>,
  fields: Vec<String>,
  params: Vec<f64>,
}

impl CompiledScript {
  pub(crate) fn evaluate(
    &self,
    fast_fields: &FastFieldsReader,
    doc_id: DocId,
    base_score: f32,
  ) -> Option<f32> {
    let mut stack: SmallVec<[f64; 16]> = SmallVec::new();
    for instr in self.instructions.iter() {
      match instr {
        Instruction::PushConst(v) => stack.push(*v),
        Instruction::PushParam(idx) => stack.push(*self.params.get(*idx)?),
        Instruction::PushField(idx) => {
          let field = self.fields.get(*idx)?;
          let value = fast_fields
            .f64_value(field, doc_id)
            .or_else(|| fast_fields.i64_value(field, doc_id).map(|v| v as f64))
            .unwrap_or(0.0);
          stack.push(value);
        }
        Instruction::PushScore => stack.push(base_score as f64),
        Instruction::Add => {
          let b = stack.pop()?;
          let a = stack.pop()?;
          let val = a + b;
          if !val.is_finite() {
            return None;
          }
          stack.push(val);
        }
        Instruction::Sub => {
          let b = stack.pop()?;
          let a = stack.pop()?;
          let val = a - b;
          if !val.is_finite() {
            return None;
          }
          stack.push(val);
        }
        Instruction::Mul => {
          let b = stack.pop()?;
          let a = stack.pop()?;
          let val = a * b;
          if !val.is_finite() {
            return None;
          }
          stack.push(val);
        }
        Instruction::Div => {
          let b = stack.pop()?;
          let a = stack.pop()?;
          if b == 0.0 {
            return None;
          }
          let val = a / b;
          if !val.is_finite() {
            return None;
          }
          stack.push(val);
        }
        Instruction::Neg => {
          let a = stack.pop()?;
          let val = -a;
          if !val.is_finite() {
            return None;
          }
          stack.push(val);
        }
      }
    }
    if stack.len() != 1 {
      return None;
    }
    let value = stack.pop().unwrap();
    if !value.is_finite() {
      return None;
    }
    Some(value as f32)
  }
}

pub(crate) fn compile_script(
  script: &str,
  params: &Option<BTreeMap<String, f64>>,
  schema: &Schema,
) -> Result<CompiledScript> {
  if script.trim().is_empty() {
    bail!("script_score script cannot be empty");
  }
  if script.len() > MAX_SCRIPT_LENGTH {
    bail!(
      "script_score script length {} exceeds max {MAX_SCRIPT_LENGTH}",
      script.len()
    );
  }
  let tokens = tokenize(script)?;
  if tokens.len() > MAX_SCRIPT_TOKENS {
    bail!(
      "script_score script is too large: {} tokens (max {MAX_SCRIPT_TOKENS})",
      tokens.len()
    );
  }
  let rpn = shunting_yard(&tokens)?;
  let mut fields: Vec<String> = Vec::new();
  let mut field_indices: BTreeMap<String, usize> = BTreeMap::new();
  let mut param_indices: BTreeMap<String, usize> = BTreeMap::new();
  let mut params_vec: Vec<f64> = Vec::new();
  if let Some(p) = params {
    for (name, value) in p.iter() {
      if !value.is_finite() {
        bail!("script_score param `{name}` must be finite");
      }
      let idx = param_indices.len();
      param_indices.insert(name.clone(), idx);
      params_vec.push(*value);
    }
  }
  let mut instructions = Vec::with_capacity(rpn.len());
  for token in rpn.into_iter() {
    match token {
      Token::Number(v) => instructions.push(Instruction::PushConst(v)),
      Token::Ident(name) => {
        if name == "_score" {
          instructions.push(Instruction::PushScore);
        } else if let Some(idx) = param_indices.get(&name) {
          instructions.push(Instruction::PushParam(*idx));
        } else {
          ensure_numeric_fast(schema, &name, "script_score")?;
          let idx = *field_indices.entry(name.clone()).or_insert_with(|| {
            let next = fields.len();
            fields.push(name.clone());
            next
          });
          instructions.push(Instruction::PushField(idx));
        }
      }
      Token::Op(op) => {
        let instr = match op {
          Op::Add => Instruction::Add,
          Op::Sub => Instruction::Sub,
          Op::Mul => Instruction::Mul,
          Op::Div => Instruction::Div,
          Op::Neg => Instruction::Neg,
        };
        instructions.push(instr);
      }
      _ => {}
    }
  }
  Ok(CompiledScript {
    instructions,
    fields,
    params: params_vec,
  })
}

fn read_number_literal(first: char, chars: &mut Peekable<Chars<'_>>) -> Result<f64> {
  let mut num = String::new();
  let mut dot_count = 0usize;
  let mut digit_count = 0usize;
  let push_char = |c: char, dot_count: &mut usize, digit_count: &mut usize, buf: &mut String| {
    if c == '.' {
      *dot_count += 1;
    } else if c.is_ascii_digit() {
      *digit_count += 1;
    }
    buf.push(c);
  };
  push_char(first, &mut dot_count, &mut digit_count, &mut num);
  if dot_count > 1 {
    bail!("invalid number literal `{num}`");
  }
  while let Some(&c) = chars.peek() {
    if c.is_ascii_digit() || c == '.' {
      chars.next();
      push_char(c, &mut dot_count, &mut digit_count, &mut num);
      if dot_count > 1 {
        bail!("invalid number literal `{num}`");
      }
    } else {
      break;
    }
  }
  if digit_count == 0 {
    bail!("invalid number literal `{num}`");
  }
  num
    .parse::<f64>()
    .map_err(|_| anyhow!("invalid number literal `{num}`"))
}

fn tokenize(input: &str) -> Result<Vec<Token>> {
  let mut tokens = Vec::new();
  let mut chars = input.chars().peekable();
  let mut expect_operand = true;
  while let Some(ch) = chars.peek().copied() {
    match ch {
      ' ' | '\t' | '\r' | '\n' => {
        chars.next();
      }
      '(' => {
        tokens.push(Token::LParen);
        chars.next();
        expect_operand = true;
      }
      ')' => {
        tokens.push(Token::RParen);
        chars.next();
        expect_operand = false;
      }
      '+' => {
        tokens.push(Token::Op(Op::Add));
        chars.next();
        expect_operand = true;
      }
      '-' => {
        chars.next();
        if expect_operand {
          if let Some(next) = chars.peek() {
            if next.is_ascii_digit() || *next == '.' {
              let first = chars.next().unwrap();
              let value = read_number_literal(first, &mut chars)?;
              tokens.push(Token::Number(-value));
              expect_operand = false;
              continue;
            }
          }
          tokens.push(Token::Op(Op::Neg));
        } else {
          tokens.push(Token::Op(Op::Sub));
        }
        expect_operand = true;
      }
      '*' => {
        tokens.push(Token::Op(Op::Mul));
        chars.next();
        expect_operand = true;
      }
      '/' => {
        tokens.push(Token::Op(Op::Div));
        chars.next();
        expect_operand = true;
      }
      ch if ch.is_ascii_digit() || ch == '.' => {
        let first = chars.next().unwrap();
        let value = read_number_literal(first, &mut chars)?;
        tokens.push(Token::Number(value));
        expect_operand = false;
      }
      ch if is_ident_start(ch) => {
        let mut ident = String::new();
        while let Some(c) = chars.peek() {
          if is_ident_continue(*c) {
            ident.push(*c);
            chars.next();
          } else {
            break;
          }
        }
        tokens.push(Token::Ident(ident));
        expect_operand = false;
      }
      _ => {
        bail!("unsupported character `{ch}` in script_score");
      }
    }
  }
  Ok(tokens)
}

fn shunting_yard(tokens: &[Token]) -> Result<Vec<Token>> {
  let mut output: Vec<Token> = Vec::new();
  let mut ops: Vec<Token> = Vec::new();
  for token in tokens.iter() {
    match token {
      Token::Number(_) | Token::Ident(_) => output.push(token.clone()),
      Token::Op(op) => {
        while let Some(Token::Op(top)) = ops.last() {
          if top.precedence() > op.precedence()
            || (top.precedence() == op.precedence() && !op.is_right_associative())
          {
            if let Some(Token::Op(popped)) = ops.pop() {
              output.push(Token::Op(popped));
            }
          } else {
            break;
          }
        }
        ops.push(token.clone());
      }
      Token::LParen => ops.push(Token::LParen),
      Token::RParen => {
        let mut found_lparen = false;
        while let Some(op_token) = ops.pop() {
          match op_token {
            Token::LParen => {
              found_lparen = true;
              break;
            }
            Token::Op(op) => output.push(Token::Op(op)),
            _ => {}
          }
        }
        if !found_lparen {
          bail!("mismatched parentheses in script_score");
        }
      }
    }
  }
  while let Some(tok) = ops.pop() {
    match tok {
      Token::Op(op) => output.push(Token::Op(op)),
      Token::LParen | Token::RParen => bail!("mismatched parentheses in script_score"),
      _ => {}
    }
  }
  Ok(output)
}

fn is_ident_start(ch: char) -> bool {
  ch.is_ascii_alphabetic() || ch == '_'
}

fn is_ident_continue(ch: char) -> bool {
  ch.is_ascii_alphanumeric() || ch == '_'
}
