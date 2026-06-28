use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;

use crate::core::types::{Scalar, VectorId};

const MAGIC_NUMBER: u8 = 0xAB;
const TYPE_INSERT: u8 = 0x01;
const TYPE_DELETE: u8 = 0x02;

/// Represents a recovered operation when we replay the log on boot.
#[derive(Debug)]
pub enum WalEntry {
    Insert { id: VectorId, vector: Vec<Scalar> },
    Delete { id: VectorId },
}

/// The Write-Ahead Log. Appends binary operations to disk before they mutate RAM.
pub struct Wal {
    file: BufWriter<File>,
}

impl Wal {
    /// Opens or creates the WAL file in strict append-only mode.
    pub fn new<P: AsRef<Path>>(path: P, _dimension: usize) -> io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;

        Ok(Self {
            file: BufWriter::new(file),
        })
    }

    /// Appends an Insert command to the log.
    pub fn append_insert(&mut self, id: VectorId, vector: &[Scalar]) -> io::Result<()> {
        self.file.write_all(&[MAGIC_NUMBER, TYPE_INSERT])?;

        self.file.write_all(&id.to_le_bytes())?;

        let byte_len = vector.len() * std::mem::size_of::<Scalar>();
        let bytes = unsafe { std::slice::from_raw_parts(vector.as_ptr() as *const u8, byte_len) };
        self.file.write_all(bytes)?;

        self.file.flush()?;

        Ok(())
    }

    /// Appends a Delete command to the log (For Path C).
    pub fn append_delete(&mut self, id: VectorId) -> io::Result<()> {
        self.file.write_all(&[MAGIC_NUMBER, TYPE_DELETE])?;
        self.file.write_all(&id.to_le_bytes())?;
        self.file.flush()?;
        Ok(())
    }
    /// Reads the entire WAL from disk and reconstructs a timeline of events.
    pub fn recover<P: AsRef<Path>>(path: P, dimension: usize) -> io::Result<Vec<WalEntry>> {
        let mut file = match File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()), // No WAL exists yet
            Err(e) => return Err(e),
        };

        let mut entries = Vec::new();
        let mut header = [0u8; 2]; // Buffer for [MAGIC, TYPE]
        let mut id_buf = [0u8; 8]; // Buffer for the VectorId

        let vector_byte_size = dimension * std::mem::size_of::<Scalar>();

        loop {
            match file.read_exact(&mut header) {
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break, // Reached end of log cleanly
                Err(e) => return Err(e),
            }

            if header[0] != MAGIC_NUMBER {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "WAL corruption: Missing Magic Number",
                ));
            }

            file.read_exact(&mut id_buf)?;
            let id = VectorId::from_le_bytes(id_buf);

            match header[1] {
                TYPE_INSERT => {
                    let mut vec_bytes = vec![0u8; vector_byte_size];
                    file.read_exact(&mut vec_bytes)?;

                    let floats: Vec<Scalar> = unsafe {
                        let ptr = vec_bytes.as_ptr() as *const Scalar;
                        std::slice::from_raw_parts(ptr, dimension).to_vec()
                    };

                    entries.push(WalEntry::Insert { id, vector: floats });
                }
                TYPE_DELETE => {
                    entries.push(WalEntry::Delete { id });
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "WAL corruption: Unknown Entry Type",
                    ));
                }
            }
        }

        Ok(entries)
    }
}
