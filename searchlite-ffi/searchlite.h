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
// Returned when a Rust panic is caught across the FFI boundary. The operation
// did not complete; callers may retry. After a panic from a mutating call,
// closing and reopening the index handle is the safest way to ensure consistency.

IndexHandle* searchlite_index_open(const char* path, bool create_if_missing);
void searchlite_index_close(IndexHandle* handle);
int32_t searchlite_add_json(IndexHandle* handle, const char* json, size_t json_len);
int32_t searchlite_add_json_batch(IndexHandle* handle, const char* json, size_t json_len);
int32_t searchlite_commit(IndexHandle* handle);
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
