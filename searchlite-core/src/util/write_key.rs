use anyhow::{anyhow, bail, Result};
use argon2::{password_hash::SaltString, Argon2, Params, PasswordHasher};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use hmac::{Hmac, Mac};
use rand::rngs::OsRng;
use sha2::Sha256;
use subtle::ConstantTimeEq;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct KdfParams {
  pub m_cost: u32,
  pub t_cost: u32,
  pub p_cost: u32,
  pub output_len: u32,
}

/// Digest + parameters stored in the manifest when a write key is configured.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct WriteKeyMeta {
  pub salt_b64: String,
  pub hash_b64: String,
  pub params: KdfParams,
}

const DEFAULT_M_COST: u32 = 19; // ~512 MiB
const DEFAULT_T_COST: u32 = 2;
const DEFAULT_P_COST: u32 = 1;
const DEFAULT_OUTPUT_LEN: u32 = 32;

pub fn default_kdf_params() -> KdfParams {
  KdfParams {
    m_cost: DEFAULT_M_COST,
    t_cost: DEFAULT_T_COST,
    p_cost: DEFAULT_P_COST,
    output_len: DEFAULT_OUTPUT_LEN,
  }
}

pub fn derive_write_key_meta(key: &str, params: Option<KdfParams>) -> Result<WriteKeyMeta> {
  if key.trim().is_empty() {
    bail!("write key cannot be empty");
  }
  let params = params.unwrap_or_else(default_kdf_params);
  let salt = SaltString::generate(&mut OsRng);
  let argon_params = Params::new(
    params.m_cost,
    params.t_cost,
    params.p_cost,
    Some(params.output_len as usize),
  )
  .map_err(|e| anyhow!("invalid Argon2 params: {e}"))?;
  let argon2 = Argon2::new(
    argon2::Algorithm::Argon2id,
    argon2::Version::V0x13,
    argon_params,
  );
  let hash = argon2
    .hash_password(key.as_bytes(), &salt)
    .map_err(|e| anyhow!("failed to hash write key: {e}"))?;
  Ok(WriteKeyMeta {
    salt_b64: salt.as_str().to_string(),
    hash_b64: STANDARD.encode(hash.hash.ok_or_else(|| anyhow!("hash missing"))?.as_bytes()),
    params,
  })
}

pub fn verify_write_key(key: &str, meta: &WriteKeyMeta) -> Result<()> {
  let salt =
    SaltString::from_b64(&meta.salt_b64).map_err(|e| anyhow!("invalid salt encoding: {e}"))?;
  let hash_bytes = STANDARD
    .decode(&meta.hash_b64)
    .map_err(|e| anyhow!("invalid hash encoding: {e}"))?;
  let params = Params::new(
    meta.params.m_cost,
    meta.params.t_cost,
    meta.params.p_cost,
    Some(meta.params.output_len as usize),
  )
  .map_err(|e| anyhow!("invalid Argon2 params: {e}"))?;
  let argon2 = Argon2::new(argon2::Algorithm::Argon2id, argon2::Version::V0x13, params);
  let candidate_hash = argon2
    .hash_password(key.as_bytes(), &salt)
    .map_err(|e| anyhow!("failed to hash candidate: {e}"))?;
  let Some(phf) = candidate_hash.hash else {
    bail!("derived hash missing");
  };
  let ok = hash_bytes.as_slice().ct_eq(phf.as_bytes()).into();
  if ok {
    Ok(())
  } else {
    bail!("invalid write key")
  }
}

/// Compute a binding token tied to the index UUID; stored alongside WAL/segment to detect manifest tampering.
pub fn binding_for_uuid(key: &str, uuid: &uuid::Uuid) -> Vec<u8> {
  let mut mac = Hmac::<Sha256>::new_from_slice(key.as_bytes()).expect("HMAC key init");
  mac.update(uuid.as_bytes());
  mac.finalize().into_bytes().to_vec()
}

/// Constant-time check of bindings.
pub fn verify_binding(expected: &[u8], candidate: &[u8]) -> bool {
  expected.ct_eq(candidate).into()
}
