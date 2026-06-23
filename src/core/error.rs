use std::fmt;

#[derive(Debug)]
pub enum ParsecError {
    DimensionMismatch { expected: usize, found: usize },
    VectorNotFound(u64),
}

impl fmt::Display for ParsecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, found } => {
                write!(
                    f,
                    "Dimenson mismatch: expected {}, but found {}",
                    expected, found
                )
            }
            Self::VectorNotFound(id) => {
                write!(f, "Vector with ID {} not found", id)
            }
        }
    }
}

impl std::error::Error for ParsecError {}

pub type Result<T> = std::result::Result<T, ParsecError>;
