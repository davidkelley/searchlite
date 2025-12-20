#ifndef SEARCHLITE_H
#define SEARCHLITE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct IndexHandle IndexHandle;

IndexHandle* searchlite_index_open(const char* path, bool create_if_missing);
void searchlite_index_close(IndexHandle* handle);
int32_t searchlite_add_json(IndexHandle* handle, const char* json, size_t json_len);
int32_t searchlite_commit(IndexHandle* handle);
size_t searchlite_search(IndexHandle* handle, const char* query, size_t limit, char* out_json_buf, size_t buf_cap);

#ifdef __cplusplus
}
#endif

#endif // SEARCHLITE_H
