#ifndef SEARCHLITE_H
#define SEARCHLITE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#if defined(_WIN32) && !defined(_SSIZE_T_DEFINED)
#  include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#  define _SSIZE_T_DEFINED
#endif
#if !defined(_WIN32) && !defined(_SSIZE_T_DEFINED)
#  include <sys/types.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct IndexHandle IndexHandle;

#define SEARCHLITE_ERR_PANIC (-100)
// Returned when a write key is required or incorrect.
#define SEARCHLITE_ERR_WRITE_KEY (-8)
// Returned when a Rust panic is caught across the FFI boundary. The operation
// did not complete; callers may retry. After a panic from a mutating call,
// closing and reopening the index handle is the safest way to ensure consistency.

IndexHandle* searchlite_index_open(const char* path, bool create_if_missing);
IndexHandle* searchlite_index_open_with_write_key(const char* path, bool create_if_missing, const char* write_key);
void searchlite_index_close(IndexHandle* handle);
int32_t searchlite_add_json(IndexHandle* handle, const char* json, size_t json_len);
int32_t searchlite_add_json_with_write_key(IndexHandle* handle, const char* json, size_t json_len, const char* write_key);
int32_t searchlite_add_json_batch(IndexHandle* handle, const char* json, size_t json_len);
int32_t searchlite_add_json_batch_with_write_key(IndexHandle* handle, const char* json, size_t json_len, const char* write_key);
int32_t searchlite_commit(IndexHandle* handle);
int32_t searchlite_commit_with_write_key(IndexHandle* handle, const char* write_key);
// Search output buffer convention (applies to `searchlite_search` and
// `searchlite_search_request`):
// - Return `0` means an error (null argument, search failure, or JSON
//   serialization failure). The buffer is untouched.
// - A positive return `N` with `N <= buf_cap - 1` means success: `N` bytes of
//   JSON were written to `out_json_buf` followed by a NUL terminator.
// - A positive return `N` with `N > buf_cap` means the buffer was too small:
//   no JSON was written (when `buf_cap >= 1` the buffer is NUL-terminated at
//   index 0), and `N` is the required size including the NUL terminator.
//   Callers should allocate `N` bytes and retry.
// - `searchlite_search` additionally returns `SEARCHLITE_ERR_PANIC` (`-100`)
//   if a Rust panic was caught.
ssize_t searchlite_search(
  IndexHandle* handle,
  const char* query,
  size_t limit,
  const char* cursor,
  const char* aggs_json,
  size_t aggs_len,
  char* out_json_buf,
  size_t buf_cap);
size_t searchlite_search_request(
  IndexHandle* handle,
  const char* request_json,
  size_t request_len,
  char* out_json_buf,
  size_t buf_cap);

#ifdef __cplusplus
}
#endif

#endif // SEARCHLITE_H
