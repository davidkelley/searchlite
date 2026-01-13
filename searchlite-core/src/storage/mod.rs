use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use parking_lot::RwLock;

pub trait StorageFile: Read + Write + Seek + Send {
  fn set_len(&mut self, len: u64) -> Result<()>;
  fn sync_all(&mut self) -> Result<()>;
}

impl StorageFile for File {
  fn set_len(&mut self, len: u64) -> Result<()> {
    File::set_len(self, len).map_err(Into::into)
  }

  fn sync_all(&mut self) -> Result<()> {
    File::sync_all(self).map_err(Into::into)
  }
}

pub type DynFile = Box<dyn StorageFile>;

pub trait Storage: Send + Sync {
  fn root(&self) -> &Path;
  fn ensure_dir(&self, path: &Path) -> Result<()>;
  fn exists(&self, path: &Path) -> bool;
  fn open_read(&self, path: &Path) -> Result<DynFile>;
  fn open_write(&self, path: &Path) -> Result<DynFile>;
  fn open_append(&self, path: &Path) -> Result<DynFile>;
  fn read_to_end(&self, path: &Path) -> Result<Vec<u8>>;
  fn write_all(&self, path: &Path, data: &[u8]) -> Result<()>;
  fn atomic_write(&self, path: &Path, data: &[u8]) -> Result<()>;
  fn remove(&self, path: &Path) -> Result<()>;
  fn remove_dir_all(&self, path: &Path) -> Result<()>;
}

pub struct FsStorage {
  root: PathBuf,
}

impl FsStorage {
  pub fn new(root: PathBuf) -> Self {
    Self { root }
  }
}

impl Storage for FsStorage {
  fn root(&self) -> &Path {
    &self.root
  }

  fn ensure_dir(&self, path: &Path) -> Result<()> {
    fs::create_dir_all(path)?;
    Ok(())
  }

  fn exists(&self, path: &Path) -> bool {
    path.exists()
  }

  fn open_read(&self, path: &Path) -> Result<DynFile> {
    Ok(Box::new(File::open(path)?))
  }

  fn open_write(&self, path: &Path) -> Result<DynFile> {
    if let Some(parent) = path.parent() {
      fs::create_dir_all(parent)?;
    }
    Ok(Box::new(File::create(path)?))
  }

  fn open_append(&self, path: &Path) -> Result<DynFile> {
    if let Some(parent) = path.parent() {
      fs::create_dir_all(parent)?;
    }
    let file = File::options()
      .create(true)
      .append(true)
      .read(true)
      .open(path)?;
    Ok(Box::new(file))
  }

  fn read_to_end(&self, path: &Path) -> Result<Vec<u8>> {
    Ok(fs::read(path)?)
  }

  fn write_all(&self, path: &Path, data: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
      fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    file.write_all(data)?;
    file.sync_all()?;
    sync_dir(path)?;
    Ok(())
  }

  fn atomic_write(&self, path: &Path, data: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    if let Some(parent) = path.parent() {
      fs::create_dir_all(parent)?;
    }
    {
      let mut file = File::create(&tmp)?;
      file.write_all(data)?;
      file.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    sync_dir(path)?;
    Ok(())
  }

  fn remove(&self, path: &Path) -> Result<()> {
    if path.exists() {
      if let Some(parent) = path.parent() {
        if !parent.exists() {
          return Ok(());
        }
      }
      fs::remove_file(path)
        .map_err(|e| anyhow!("failed to remove file {}: {e}", path.display()))?;
    }
    Ok(())
  }

  fn remove_dir_all(&self, path: &Path) -> Result<()> {
    if path.exists() {
      fs::remove_dir_all(path)
        .map_err(|e| anyhow!("failed to remove directory {}: {e}", path.display()))?;
    }
    Ok(())
  }
}

fn sync_dir(path: &Path) -> Result<()> {
  if let Some(parent) = path.parent() {
    let dir = File::open(parent)?;
    dir.sync_all()?;
  }
  Ok(())
}

pub struct InMemoryStorage {
  root: PathBuf,
  files: RwLock<HashMap<PathBuf, Arc<RwLock<Vec<u8>>>>>,
}

impl InMemoryStorage {
  pub fn new(root: PathBuf) -> Self {
    Self {
      root,
      files: RwLock::new(HashMap::new()),
    }
  }

  fn entry(&self, path: &Path) -> Arc<RwLock<Vec<u8>>> {
    let mut map = self.files.write();
    map
      .entry(path.to_path_buf())
      .or_insert_with(|| Arc::new(RwLock::new(Vec::new())))
      .clone()
  }

  fn open_with_mode(&self, path: &Path, truncate: bool, append: bool) -> Result<DynFile> {
    let data = self.entry(path);
    if truncate {
      data.write().clear();
    }
    let pos = if append { data.read().len() as u64 } else { 0 };
    Ok(Box::new(MemFile { data, pos }))
  }
}

impl Storage for InMemoryStorage {
  fn root(&self) -> &Path {
    &self.root
  }

  fn ensure_dir(&self, _path: &Path) -> Result<()> {
    Ok(())
  }

  fn exists(&self, path: &Path) -> bool {
    self.files.read().contains_key(path)
  }

  fn open_read(&self, path: &Path) -> Result<DynFile> {
    if !self.exists(path) {
      return Err(anyhow!("file {:?} missing in memory storage", path));
    }
    self.open_with_mode(path, false, false)
  }

  fn open_write(&self, path: &Path) -> Result<DynFile> {
    self.open_with_mode(path, true, false)
  }

  fn open_append(&self, path: &Path) -> Result<DynFile> {
    self.open_with_mode(path, false, true)
  }

  fn read_to_end(&self, path: &Path) -> Result<Vec<u8>> {
    if let Some(buf) = self.files.read().get(path) {
      return Ok(buf.read().clone());
    }
    Err(anyhow!("file {:?} missing in memory storage", path))
  }

  fn write_all(&self, path: &Path, data: &[u8]) -> Result<()> {
    let entry = self.entry(path);
    let mut guard = entry.write();
    guard.clear();
    guard.extend_from_slice(data);
    Ok(())
  }

  fn atomic_write(&self, path: &Path, data: &[u8]) -> Result<()> {
    self.write_all(path, data)
  }

  fn remove(&self, path: &Path) -> Result<()> {
    let mut map = self.files.write();
    map.remove(path);
    Ok(())
  }

  fn remove_dir_all(&self, path: &Path) -> Result<()> {
    let mut map = self.files.write();
    map.retain(|p, _| !p.starts_with(path));
    Ok(())
  }
}

struct MemFile {
  data: Arc<RwLock<Vec<u8>>>,
  pos: u64,
}

impl Read for MemFile {
  fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
    let data = self.data.read();
    if self.pos as usize >= data.len() {
      return Ok(0);
    }
    let available = data.len() - self.pos as usize;
    let len = available.min(buf.len());
    buf[..len].copy_from_slice(&data[self.pos as usize..self.pos as usize + len]);
    self.pos += len as u64;
    Ok(len)
  }
}

impl Write for MemFile {
  fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
    let mut data = self.data.write();
    let end = (self.pos as usize).saturating_add(buf.len());
    if end > data.len() {
      data.resize(end, 0);
    }
    data[self.pos as usize..end].copy_from_slice(buf);
    self.pos = end as u64;
    Ok(buf.len())
  }

  fn flush(&mut self) -> std::io::Result<()> {
    Ok(())
  }
}

impl Seek for MemFile {
  fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
    let new = match pos {
      SeekFrom::Start(off) => off as i64,
      SeekFrom::End(off) => {
        let len = self.data.read().len() as i64;
        len + off
      }
      SeekFrom::Current(off) => self.pos as i64 + off,
    };
    if new < 0 {
      return Err(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        "negative seek",
      ));
    }
    self.pos = new as u64;
    Ok(self.pos)
  }
}

impl StorageFile for MemFile {
  fn set_len(&mut self, len: u64) -> Result<()> {
    let mut data = self.data.write();
    data.resize(len as usize, 0);
    if self.pos > len {
      self.pos = len;
    }
    Ok(())
  }

  fn sync_all(&mut self) -> Result<()> {
    Ok(())
  }
}
