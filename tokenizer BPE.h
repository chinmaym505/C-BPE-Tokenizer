#pragma once
#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sched.h>
#endif
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <limits.h>
#include <sys/stat.h>
#include <pthread.h>
#include "libs/cJSON/cJSON.h" // This includes the cJSON implementation directly
#define NUM_CORES 12
#define THREADS_PER_CORE 1
#define TOTAL_BATCHES (NUM_CORES * THREADS_PER_CORE)

//#define TESTING // Comment this out to disable debug prints

/* Simple open-addressing hash map string -> int for vocab and merges */
typedef struct {
    char *key;
    int value;
} MapEntry;

typedef struct {
    MapEntry *entries;
    size_t capacity;
    size_t count;
} StrIntMap;

typedef struct { uint64_t *keys; int *values; size_t capacity; size_t count; } Map64;

static inline unsigned long str_hash(const char *s) {
    unsigned long hash = 5381;
    int c;
    while ((c = *s++)) hash = ((hash << 5) + hash) + (unsigned char)c;
    return hash;
}

static StrIntMap *map_create(size_t cap) {
    StrIntMap *m = malloc(sizeof(StrIntMap));
    if (!m) return NULL;
    size_t c = 1;
    while (c < cap) c <<= 1;
    m->capacity = c;
    m->count = 0;
    m->entries = calloc(m->capacity, sizeof(MapEntry));
    if (!m->entries) { free(m); return NULL; }
    return m;
}

static void map_free(StrIntMap *m) {
    if (!m) return;
    for (size_t i = 0; i < m->capacity; i++) if (m->entries[i].key) free(m->entries[i].key);
    free(m->entries);
    free(m);
}

static int map_get(StrIntMap *m, const char *key) {
    if (!m || !key) return -1;
    unsigned long h = str_hash(key);
    size_t idx = h & (m->capacity - 1);
    for (size_t probe = 0; probe < m->capacity; probe++) {
        MapEntry *e = &m->entries[idx];
        if (!e->key) return -1;
        if (strcmp(e->key, key) == 0) return e->value;
        idx = (idx + 1) & (m->capacity - 1);
    }
    return -1;
}

static int map_put(StrIntMap *m, const char *key, int value) {
    if (!m || !key) return -1;
    if (m->count * 2 >= m->capacity) {
        /* resize */
        size_t newcap = m->capacity << 1;
        MapEntry *newentries = calloc(newcap, sizeof(MapEntry));
        if (!newentries) return -1;
        for (size_t i = 0; i < m->capacity; i++) {
            if (m->entries[i].key) {
                unsigned long h = str_hash(m->entries[i].key);
                size_t idx = h & (newcap - 1);
                while (newentries[idx].key) idx = (idx + 1) & (newcap - 1);
                newentries[idx].key = m->entries[i].key; // move ownership
                newentries[idx].value = m->entries[i].value;
            }
        }
        free(m->entries);
        m->entries = newentries;
        m->capacity = newcap;
    }
    unsigned long h = str_hash(key);
    size_t idx = h & (m->capacity - 1);
    while (m->entries[idx].key) idx = (idx + 1) & (m->capacity - 1);
    m->entries[idx].key = strdup(key);
    if (!m->entries[idx].key) return -1;
    m->entries[idx].value = value;
    m->count++;
    return 0;
}
typedef struct
{
    char *str1;
    char *str2;
    int rank;
} merge_pair;
typedef struct
{
    char *token;
    int id;
} tokenID;

typedef struct Symbol {
    char *s;
    unsigned long h;
    int len;
    struct Symbol *prev;
    struct Symbol *next;
    int alive;
} Symbol;

typedef struct {
    Symbol *left;
    int rank;
} Pair;

typedef struct {
    Pair *data;
    int size;
    int cap;
} MinHeap;

static void heap_init(MinHeap *h, int cap) {
    h->data = NULL; h->size = 0; h->cap = 0;
    if (cap > 0) { h->data = malloc(cap * sizeof(Pair)); if (h->data) h->cap = cap; }
}

static void heap_push(MinHeap *h, Pair p) {
    if (!h) return;
    if (h->size == h->cap) {
        int nc = h->cap ? h->cap * 2 : 4;
        Pair *nd = realloc(h->data, nc * sizeof(Pair));
        if (!nd) return;
        h->data = nd; h->cap = nc;
    }
    int i = h->size++;
    h->data[i] = p;
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (h->data[parent].rank <= h->data[i].rank) break;
        Pair tmp = h->data[parent]; h->data[parent] = h->data[i]; h->data[i] = tmp;
        i = parent;
    }
}

static int heap_pop(MinHeap *h, Pair *out) {
    if (!h || h->size == 0) return 0;
    *out = h->data[0];
    h->size--;
    if (h->size > 0) h->data[0] = h->data[h->size];
    int i = 0;
    while (1) {
        int l = 2 * i + 1, r = 2 * i + 2, smallest = i;
        if (l < h->size && h->data[l].rank < h->data[smallest].rank) smallest = l;
        if (r < h->size && h->data[r].rank < h->data[smallest].rank) smallest = r;
        if (smallest == i) break;
        Pair tmp = h->data[i]; h->data[i] = h->data[smallest]; h->data[smallest] = tmp;
        i = smallest;
    }
    return 1;
}

static void heap_free(MinHeap *h) { if (!h) return; if (h->data) free(h->data); h->data = NULL; h->size = 0; h->cap = 0; }

/* Per-batch pool for arena allocations */
typedef struct {
    char *arena;
    size_t arena_size;
    size_t arena_off;
    tokenID *token_structs; /* array of tokenID structs for the batch */
    int max_tokens;
    char **malloced_strings;
    int malloced_count;
    int malloced_cap;
    /* reusable scratch buffers to avoid per-word malloc/free */
    Symbol *symbols_buf;
    int symbols_cap;
    Pair *pair_buf;
    int pair_cap;
} BatchPool;

typedef struct
{
    tokenID **vocab;
    int vocab_size;
    merge_pair *merges;
    int merge_size;
    StrIntMap *vocab_map;
    /* merges map: use 64-bit composite hash to avoid separator assumptions */
    Map64 *merges_map64;
    /* reverse map id -> token string for O(1) decode */
    char **id_to_token;
    int id_map_capacity;
    /* runtime scratch space */
    BatchPool **last_pools; /* used when returning pooled results */
    int last_pool_count;
} BPEtokenizer;

// Helper function to split text into tokens using whitespace only
char **split_string_words(const char *input, int *out_count) {
    if (!input) return NULL;

    /* Single buffer parse: allocate one copy and return pointers into it. */
    char *buf = strdup(input);
    if (!buf) return NULL;

    int capacity = 64;
    int count = 0;
    char **arr = malloc(capacity * sizeof(char *));
    if (!arr) { free(buf); return NULL; }

    char *p = buf;
    while (*p) {
        /* skip whitespace */
        while (*p && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')) p++;
        if (!*p) break;
        if (count >= capacity) {
            capacity *= 2;
            char **new_arr = realloc(arr, capacity * sizeof(char *));
            if (!new_arr) { free(arr); free(buf); return NULL; }
            arr = new_arr;
        }
        arr[count++] = p;
        /* advance to next whitespace and terminate current token */
        while (*p && *p != ' ' && *p != '\n' && *p != '\t' && *p != '\r') p++;
        if (*p) { *p = '\0'; p++; }
    }

    /* Null-terminate the pointer array and return buffer via arr[-1] trick not used; instead return arr and expect free_string_array to free both */
    if (count == 0) {
        free(arr);
        free(buf);
        if (out_count) *out_count = 0;
        return NULL;
    }

    if (count >= capacity) {
        char **new_arr = realloc(arr, (capacity + 1) * sizeof(char*));
        if (new_arr) arr = new_arr;
    }
    arr[count] = NULL;
    if (out_count) *out_count = count;

    /* Store the buffer pointer just before the array to free both together later. Allocate a single block: [buf_ptr][arr pointers...]
       We'll return pointer to combined[1] so caller sees token pointers at index 0. */
    char **combined = malloc((1 + count + 1) * sizeof(char*));
    if (!combined) { free(arr); free(buf); return NULL; }
    combined[0] = buf; // store buffer
    for (int i = 0; i < count; i++) combined[i + 1] = arr[i];
    combined[count + 1] = NULL;
    free(arr);
    return &combined[1];
}

// Helper function to free the temporary string array
void free_string_array(char **arr)
{
    if (!arr) return;
    /* arr was returned as combined+1, so combined = arr-1 */
    char **combined = arr - 1;
    char *buf = combined[0];
    free(buf);
    free(combined);
}

/* compare_merge_pairs removed: bsearch-based lookup replaced by Map64 hash map */

static int find_merge_rank(BPEtokenizer *tokenizer, const char *first, const char *second)
{
    if (!tokenizer || !tokenizer->merges || !first || !second) return -1;
    /* composite 64-bit hash */
    unsigned long h1 = str_hash(first);
    unsigned long h2 = str_hash(second);
    uint64_t key = ((uint64_t)h1 << 32) ^ (uint64_t)h2;
    if (!tokenizer->merges_map64) return -1;
    /* lookup in merges_map64 */
    Map64 *m = tokenizer->merges_map64;
    if (m->count == 0) return -1;
    size_t idx = (size_t)(key & (m->capacity - 1));
    for (size_t probe = 0; probe < m->capacity; probe++) {
        if (m->keys[idx] == 0) return -1;
        if (m->keys[idx] == key) return m->values[idx];
        idx = (idx + 1) & (m->capacity - 1);
    }
    return -1;
}

/* faster lookup when caller already has precomputed hashes for both strings */
static int find_merge_rank_by_hash(BPEtokenizer *tokenizer, unsigned long h1, unsigned long h2)
{
    if (!tokenizer || !tokenizer->merges_map64) return -1;
    uint64_t key = ((uint64_t)h1 << 32) ^ (uint64_t)h2;
    Map64 *m = tokenizer->merges_map64;
    if (!m || m->count == 0) return -1;
    size_t idx = (size_t)(key & (m->capacity - 1));
    for (size_t probe = 0; probe < m->capacity; probe++) {
        if (m->keys[idx] == 0) return -1;
        if (m->keys[idx] == key) return m->values[idx];
        idx = (idx + 1) & (m->capacity - 1);
    }
    return -1;
}

static char *concat_strings(const char *a, const char *b)
{
    size_t len_a = strlen(a);
    size_t len_b = strlen(b);
    char *out = malloc(len_a + len_b + 1);
    if (!out)
        return NULL;
    memcpy(out, a, len_a);
    memcpy(out + len_a, b, len_b);
    out[len_a + len_b] = '\0';
    return out;
}

int find_token_id(BPEtokenizer *tokenizer, const char *token)
{
    if (!tokenizer || !token)
        return -1;
    if (tokenizer->vocab_map) {
        return map_get(tokenizer->vocab_map, token);
    }
    for (int i = 0; i < tokenizer->vocab_size; i++) {
        if (tokenizer->vocab[i] && tokenizer->vocab[i]->token && strcmp(tokenizer->vocab[i]->token, token) == 0) {
            return tokenizer->vocab[i]->id;
        }
    }
    return -1;
}
char *find_token_by_id(BPEtokenizer *tokenizer, int id)
{
    if (!tokenizer) return NULL;
    if (tokenizer->id_to_token && id >= 0 && id < tokenizer->id_map_capacity) {
        return tokenizer->id_to_token[id];
    }
    for (int i = 0; i < tokenizer->vocab_size; i++)
    {
        if (tokenizer->vocab[i] && tokenizer->vocab[i]->id == id)
        {
            return tokenizer->vocab[i]->token;
        }
    }
    return NULL;
}
// Function to free all allocated memory for the tokenizer
void freeTokenizer(BPEtokenizer *tokenizer)
{
    if (!tokenizer)
        return;
    if (tokenizer->vocab_map) {
        map_free(tokenizer->vocab_map);
        tokenizer->vocab_map = NULL;
    }
    if (tokenizer->merges_map64) {
        if (tokenizer->merges_map64->keys) free(tokenizer->merges_map64->keys);
        if (tokenizer->merges_map64->values) free(tokenizer->merges_map64->values);
        free(tokenizer->merges_map64);
        tokenizer->merges_map64 = NULL;
    }
    if (tokenizer->vocab)
    {
        for (int i = 0; i < tokenizer->vocab_size; i++)
        {
            if (tokenizer->vocab[i])
            {
                free(tokenizer->vocab[i]->token);
                free(tokenizer->vocab[i]);
            }
        }
        free(tokenizer->vocab);
        tokenizer->vocab = NULL;
    }
    if (tokenizer->id_to_token) {
        free(tokenizer->id_to_token);
        tokenizer->id_to_token = NULL;
        tokenizer->id_map_capacity = 0;
    }
    if (tokenizer->merges)
    {
        for (int i = 0; i < tokenizer->merge_size; i++)
        {
            free(tokenizer->merges[i].str1);
            free(tokenizer->merges[i].str2);
        }
        free(tokenizer->merges);
        tokenizer->merges = NULL;
        tokenizer->merge_size = 0;
    }
}

static void *build_token_tree(BPEtokenizer *tokenizer, int degree)
{
    (void)tokenizer; (void)degree; return NULL; /* removed: now using map */
}

// Function to initialize the tokenizer from a JSON file
BPEtokenizer initTokenizer(char *json_path)
{
    #ifdef TESTING
    printf("RUNNING TOKENIZER IN TEST MODE WITH DEBUG PRINTS ENABLED.\n");
    #endif
    BPEtokenizer t = {0};
    FILE *fp = fopen(json_path, "r");
    if (fp == NULL)
    {
        perror("Error: Unable to open the tokenizer JSON file.\n");
        return t;
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buffer = malloc(fsize + 1);
    if (!buffer)
    {
        perror("Error: Memory allocation failed.\n");
        fclose(fp);
        return t;
    }
    fread(buffer, 1, fsize, fp);
    fclose(fp);
    buffer[fsize] = '\0';
    cJSON *json = cJSON_Parse(buffer);
    free(buffer);
    if (json == NULL)
    {
        const char *error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL)
        {
            fprintf(stderr, "Error: %s\n", error_ptr);
        }
        return t;
    }
    cJSON *model = cJSON_GetObjectItemCaseSensitive(json, "model");
    if (!cJSON_IsObject(model))
    {
        fprintf(stderr, "Error: Invalid model format in tokenizer JSON.\n");
        cJSON_Delete(json);
        return t;
    }
    cJSON *vocab = cJSON_GetObjectItemCaseSensitive(model, "vocab");
    if (!cJSON_IsObject(vocab))
    {
        fprintf(stderr, "Error: Invalid vocab format in tokenizer JSON.\n");
        cJSON_Delete(json);
        return t;
    }
    // Count the number of vocab items
    int count = 0;
    cJSON *temp = vocab->child;
    while (temp) {
        count++;
        temp = temp->next;
    }
    t.vocab_size = count;
    t.vocab = calloc(t.vocab_size, sizeof(tokenID *));
    if (!t.vocab)
    {
        fprintf(stderr, "Error: Memory allocation failed for vocab array.\n");
        cJSON_Delete(json);
        return t;
    }
    cJSON *vocab_item = vocab->child;
    int index = 0;
    int max_id = -1;
    while (vocab_item)
    {
        tokenID *ti = malloc(sizeof(tokenID));
        if (!ti)
        {
            fprintf(stderr, "Error: Memory allocation failed for tokenID.\n");
            cJSON_Delete(json);
            freeTokenizer(&t);
            return t;
        }
        ti->token = strdup(vocab_item->string);
        ti->id = vocab_item->valueint;
        if (ti->id > max_id) max_id = ti->id;
        t.vocab[index++] = ti;
        vocab_item = vocab_item->next;
    }

    /* Build vocab map for O(1) lookup */
    t.vocab_map = map_create(t.vocab_size * 2 + 1);
    if (t.vocab_map) {
        for (int i = 0; i < t.vocab_size; i++) if (t.vocab[i]) map_put(t.vocab_map, t.vocab[i]->token, t.vocab[i]->id);
    }

    /* Build reverse id->token array for O(1) decode (if ids are reasonably dense).
       Defend against malicious or malformed JSON by clamping the allowed id range. */
    if (max_id >= 0) {
        size_t requested = (size_t)max_id + 1;
        /* allow a small multiplier over vocab_size to tolerate sparse ids, but cap the allocation */
        size_t per_vocab_limit = (size_t)t.vocab_size * 8;
        if (per_vocab_limit < 1024) per_vocab_limit = 1024;
        const size_t HARD_LIMIT = 5000000; /* 5M entries max (~40MB on 64-bit) */
        /* use the smaller of per_vocab_limit and HARD_LIMIT so small vocabs get a proportionate cap */
        size_t max_allowed = per_vocab_limit < HARD_LIMIT ? per_vocab_limit : HARD_LIMIT;
        if (requested > (size_t)INT_MAX || requested > max_allowed) {
            fprintf(stderr, "Warning: id range too large (%zu), skipping id->token map; decode will be slower.\n", requested);
            t.id_to_token = NULL;
            t.id_map_capacity = 0;
        } else {
            t.id_map_capacity = (int)requested;
            t.id_to_token = calloc(requested, sizeof(char*));
            if (!t.id_to_token) {
                fprintf(stderr, "Warning: Failed to allocate id->token map; decode will be slower.\n");
                t.id_map_capacity = 0;
                t.id_to_token = NULL;
            } else {
                for (int i = 0; i < t.vocab_size; i++) if (t.vocab[i] && t.vocab[i]->id >=0 && t.vocab[i]->id < t.id_map_capacity) t.id_to_token[t.vocab[i]->id] = t.vocab[i]->token;
            }
        }
    } else {
        t.id_to_token = NULL; t.id_map_capacity = 0;
    }

    cJSON *merges = cJSON_GetObjectItemCaseSensitive(model, "merges");
    if (cJSON_IsArray(merges))
    {
        t.merge_size = cJSON_GetArraySize(merges);
        t.merges = malloc(t.merge_size * sizeof(merge_pair));
        if (!t.merges)
        {
            fprintf(stderr, "Error: Memory allocation failed for merges array.\n");
            cJSON_Delete(json);
            freeTokenizer(&t);
            return t;
        }
        int merge_index = 0;
        cJSON *merge_item = NULL;
        cJSON_ArrayForEach(merge_item, merges)
        {
            if (cJSON_IsArray(merge_item) && cJSON_GetArraySize(merge_item) == 2)
            {
                cJSON *first = cJSON_GetArrayItem(merge_item, 0);
                cJSON *second = cJSON_GetArrayItem(merge_item, 1);
                if (cJSON_IsString(first) && cJSON_IsString(second))
                {
                    t.merges[merge_index].str1 = strdup(first->valuestring);
                    t.merges[merge_index].str2 = strdup(second->valuestring);
                    t.merges[merge_index].rank = merge_index;
                    if (!t.merges[merge_index].str1 || !t.merges[merge_index].str2)
                    {
                        fprintf(stderr, "Error: Memory allocation failed for merge strings.\n");
                        cJSON_Delete(json);
                        freeTokenizer(&t);
                        return t;
                    }
                    merge_index++;
                }
            }
        }
        t.merge_size = merge_index;
            if (t.merge_size > 0)
            {
                /* build 64-bit hash map for merges */
                size_t cap = 1;
                while (cap < (size_t)(t.merge_size * 2 + 1)) cap <<= 1;
                t.merges_map64 = malloc(sizeof(*t.merges_map64));
                if (t.merges_map64) {
                    t.merges_map64->keys = calloc(cap, sizeof(uint64_t));
                    t.merges_map64->values = malloc(cap * sizeof(int));
                    if (!t.merges_map64->keys || !t.merges_map64->values) {
                        fprintf(stderr, "Warning: Failed to allocate merges_map64; continuing without fast merges lookup.\n");
                        if (t.merges_map64->keys) free(t.merges_map64->keys);
                        if (t.merges_map64->values) free(t.merges_map64->values);
                        free(t.merges_map64);
                        t.merges_map64 = NULL;
                    } else {
                        t.merges_map64->capacity = cap;
                        t.merges_map64->count = 0;
                        for (int mi = 0; mi < t.merge_size; mi++) {
                            unsigned long h1 = str_hash(t.merges[mi].str1);
                            unsigned long h2 = str_hash(t.merges[mi].str2);
                            uint64_t key = ((uint64_t)h1 << 32) ^ (uint64_t)h2;
                            size_t idx = (size_t)(key & (cap - 1));
                            while (t.merges_map64->keys[idx] != 0) idx = (idx + 1) & (cap - 1);
                            t.merges_map64->keys[idx] = key;
                            t.merges_map64->values[idx] = t.merges[mi].rank;
                            t.merges_map64->count++;
                        }
                    }
                } else {
                    fprintf(stderr, "Warning: Failed to allocate merges_map64 struct; continuing without fast merges lookup.\n");
                    t.merges_map64 = NULL;
                }
            }
    }

    cJSON_Delete(json);
    t.last_pools = NULL;
    t.last_pool_count = 0;
    return t;
}

// Function to free the array of tokenID pointers
void free_tokenID_array(tokenID **arr)
{
    if (!arr)
        return;
    /* check for wrapper header stored at arr[-1] with low-bit marker */
    void **combined = (void**)arr - 1;
    if (((uintptr_t)combined[0] & 1) != 0) {
        /* pooled result path */
        typedef struct { int pool_count; BatchPool **pools; } TokenListWrapper;
        TokenListWrapper *wrapper = (TokenListWrapper*)((uintptr_t)combined[0] & ~((uintptr_t)1));
        if (wrapper) {
            for (int p = 0; p < wrapper->pool_count; p++) {
                    BatchPool *pool = wrapper->pools[p];
                    if (!pool) continue;
                    if (pool->token_structs) free(pool->token_structs);
                    if (pool->arena) free(pool->arena);
                    if (pool->malloced_strings) {
                        for (int m = 0; m < pool->malloced_count; m++) if (pool->malloced_strings[m]) free(pool->malloced_strings[m]);
                        free(pool->malloced_strings);
                    }
                    if (pool->symbols_buf) free(pool->symbols_buf);
                    if (pool->pair_buf) free(pool->pair_buf);
                    free(pool);
                }
            if (wrapper->pools) free(wrapper->pools);
            free(wrapper);
        }
        free(combined);
        return;
    }
    /* legacy per-token mallocs */
    for (int i = 0; arr[i] != NULL; i++)
    {
        free(arr[i]->token);
        free(arr[i]);
    }
    free(arr);
}
typedef struct {
    BPEtokenizer *tokenizer;
    tokenID ***batch_results;
    int batch_index;
    int *main_out_token_count;
    int **batch_token_counts;
    char **words_slice; // Pointer to the start of the batch's words
    int num_words_in_batch; // Number of words in this batch
    int target_core; // Core to which this thread should be assigned
    BatchPool *pool; /* per-batch pool allocated by caller */
} TokenizeBatchArgs;

void *tokenizeBatch(BPEtokenizer *tokenizer, char **words_split, int num_words_in_batch, 
                    tokenID ***batch_results, int batch_index, 
                    int *main_out_token_count, int **batch_token_counts, BatchPool *pool) {
    #ifdef TESTING
    printf("Starting tokenization for batch %d\n", batch_index);
    #endif
    int out_token_count;
    if (!tokenizer || !words_split)
        return NULL;

    /* Upper bound: a word of n chars produces at most n tokens */
    int max_tokens = 0;
    size_t total_chars = 0;
    for (int i = 0; i < num_words_in_batch; i++) { int l = strlen(words_split[i]); max_tokens += l; total_chars += (size_t)l + 1; }

    tokenID **tokens_list = malloc((max_tokens + 1) * sizeof(tokenID *));
    if (!tokens_list) {
        fprintf(stderr, "Error: Memory allocation failed for tokens list.\n");
        return NULL;
    }
    int current_token_count = 0;

    /* Use per-batch pool provided by caller */
    if (!pool) {
        (*batch_results) = NULL;
        (*batch_token_counts)[batch_index] = 0;
        return NULL;
    }

    for (int i = 0; i < num_words_in_batch; i++)
    {
        char *word = words_split[i];
        int word_len = strlen(word);
        if (word_len == 0)
            continue;

        // Single arena allocation for all initial 1-char symbols — cache hash+len per symbol
        Symbol *symbols = NULL;
        if (pool->symbols_buf && pool->symbols_cap >= word_len) {
            symbols = pool->symbols_buf;
        } else {
            symbols = malloc((word_len + 1) * sizeof(Symbol));
        }
        if (!symbols) { free(tokens_list); (*batch_results) = NULL; (*batch_token_counts)[batch_index] = 0; return NULL; }
        int symbol_count = word_len;
        for (int pos = 0; pos < word_len; pos++)
        {
            char *s = pool->arena + pool->arena_off;
            s[0] = word[pos]; s[1] = '\0';
            pool->arena_off += 2;
            symbols[pos].s = s;
            symbols[pos].len = 1;
            symbols[pos].h = str_hash(s);
            symbols[pos].prev = (pos > 0) ? &symbols[pos - 1] : NULL;
            symbols[pos].next = (pos + 1 < word_len) ? &symbols[pos + 1] : NULL;
            symbols[pos].alive = 1;
        }

        /* Build a min-heap of candidate merges (left symbol pointer + rank). Use a linked list of symbols
           so merging only affects local neighbors. This yields ~O(L log L) behavior for long words. */
        MinHeap heap;
        /* decide up front whether we'll use the pool's pair buffer for this word */
        int used_pool_pairs = (pool->pair_buf && pool->pair_cap >= symbol_count) ? 1 : 0;
        if (used_pool_pairs) {
            heap.data = pool->pair_buf;
            heap.size = 0;
            heap.cap = pool->pair_cap;
        } else {
            heap_init(&heap, symbol_count);
        }
        for (int j = 0; j < symbol_count - 1; j++) {
            int rank = find_merge_rank_by_hash(tokenizer, symbols[j].h, symbols[j + 1].h);
            if (rank >= 0) heap_push(&heap, (Pair){&symbols[j], rank});
        }

        while (heap.size > 0) {
            Pair p;
            if (!heap_pop(&heap, &p)) break;
            Symbol *left = p.left;
            if (!left || !left->alive) continue;
            Symbol *right = left->next;
            if (!right || !right->alive) continue;
            /* verify rank is still current */
            int current_rank = find_merge_rank_by_hash(tokenizer, left->h, right->h);
            if (current_rank != p.rank) {
                if (current_rank >= 0) heap_push(&heap, (Pair){left, current_rank});
                continue;
            }
            /* perform merge of left and right */
            size_t l1 = (size_t)left->len;
            size_t l2 = (size_t)right->len;
            char *merged = NULL;
            if (pool->arena_off + l1 + l2 + 1 <= pool->arena_size) {
                merged = pool->arena + pool->arena_off;
                memcpy(merged, left->s, l1);
                memcpy(merged + l1, right->s, l2);
                merged[l1 + l2] = '\0';
                pool->arena_off += l1 + l2 + 1;
            } else {
                merged = malloc(l1 + l2 + 1);
                if (!merged) break;
                memcpy(merged, left->s, l1);
                memcpy(merged + l1, right->s, l2);
                merged[l1 + l2] = '\0';
                if (pool->malloced_count + 1 > pool->malloced_cap) {
                    int newcap = pool->malloced_cap ? pool->malloced_cap * 2 : 16;
                    char **newarr = realloc(pool->malloced_strings, newcap * sizeof(char*));
                    if (!newarr) { free(merged); break; }
                    pool->malloced_strings = newarr; pool->malloced_cap = newcap;
                }
                pool->malloced_strings[pool->malloced_count++] = merged;
            }
            left->s = merged;
            left->len = (int)(l1 + l2);
            left->h = str_hash(merged);
            /* unlink right */
            Symbol *rnext = right->next;
            left->next = rnext;
            if (rnext) rnext->prev = left;
            right->alive = 0;
            symbol_count--;
            /* push updated neighboring pairs involving left */
            if (left->prev && left->prev->alive) {
                int rr = find_merge_rank_by_hash(tokenizer, left->prev->h, left->h);
                if (rr >= 0) heap_push(&heap, (Pair){left->prev, rr});
            }
            if (left->next && left->next->alive) {
                int rr = find_merge_rank_by_hash(tokenizer, left->h, left->next->h);
                if (rr >= 0) heap_push(&heap, (Pair){left, rr});
            }
        }
        /* After the merge loop, if we used the pool buffer and the heap grew (realloc moved it),
           adopt the heap's data pointer and new capacity into the pool for reuse. Otherwise,
           free the heap if it was allocated dynamically. */
        if (used_pool_pairs) {
            if (heap.data != pool->pair_buf) {
                /* heap.data points to the (possibly larger) realloc'd block. Adopt it. */
                pool->pair_buf = heap.data;
                pool->pair_cap = heap.cap;
            }
            /* Note: do not free pool->pair_buf here. If realloc moved, the original buffer was
               already freed by realloc; heap.data is the valid block and now stored in pool. */
        } else {
            heap_free(&heap);
        }

        /* symbols currently point into pool->arena; no strdup needed */

        /* Walk the linked list from the head to collect surviving symbols in order. */
        Symbol *cur = &symbols[0];
        while (cur) {
            if (cur->alive) {
                int token_id = find_token_id(tokenizer, cur->s);
                tokenID *new_token = &pool->token_structs[current_token_count];
                new_token->id = token_id;
                new_token->token = cur->s;
                tokens_list[current_token_count] = new_token;
                current_token_count++;
            }
            cur = cur->next;
        }
        /* free symbols only if it was dynamically allocated per-word */
        if (!(pool->symbols_buf && pool->symbols_cap >= word_len)) free(symbols);
    }

    if (tokens_list) tokens_list[current_token_count] = NULL;
    out_token_count = current_token_count;
    (*batch_results) = tokens_list;
    (*batch_token_counts)[batch_index] = out_token_count;
    /* store pool pointer into tokenizer->last_pools in a thread-safe way? Not safe here; caller will collect pools after join. */
    return NULL;
}

void* tokenizeBatchWrapper(void* arg) {
    TokenizeBatchArgs* args = (TokenizeBatchArgs*)arg;
#ifdef _WIN32
    HANDLE current_thread = GetCurrentThread();
    DWORD_PTR affinity_mask = (DWORD_PTR)1 << args->target_core;
    SetThreadAffinityMask(current_thread, affinity_mask);
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(args->target_core, &cpuset);
    pthread_t self = pthread_self();
    pthread_setaffinity_np(self, sizeof(cpu_set_t), &cpuset);
#endif

    tokenizeBatch(args->tokenizer, args->words_slice, args->num_words_in_batch, args->batch_results,
                  args->batch_index, args->main_out_token_count, args->batch_token_counts, args->pool);
    
    return NULL; 
}

/**
 * Returns an array of pointers to tokenID structs.
 * Should free the returned array using free_tokenID_array().
 */
tokenID **tokenize(BPEtokenizer *tokenizer, const char *text, int *out_token_count)
{
    if (!tokenizer || !text || !out_token_count){
        printf("Invalid arguments to tokenize function.\n");
        return NULL;
    }
    tokenID **tokens_list = NULL;
    int current_token_count = 0;
    int num_words = 0;
    char **words_split = split_string_words(text, &num_words);

    if (!words_split)
    {
        printf("No words found in the input text.\n");
        *out_token_count = 0;
        return NULL;
    }
    #ifdef TESTING
    printf("Number of words to tokenize: %d\n", num_words);
    #endif

    
    int batch_size = (num_words + TOTAL_BATCHES - 1) / TOTAL_BATCHES; // Dynamically calculate batch size based on total words and desired batches
    #ifdef TESTING
    printf("Calculated batch size: %d\n", batch_size);
    printf("Total batches to process: %d\n", TOTAL_BATCHES);
    #endif
    int actual_batches = (num_words + batch_size - 1) / batch_size;
    if (actual_batches <= 0) {
        free_string_array(words_split);
        *out_token_count = 0;
        return NULL;
    }

    pthread_t *thread_list = malloc(actual_batches * sizeof(pthread_t));
    tokenID ***batch_results = malloc(actual_batches * sizeof(tokenID **)); 
    int *batch_token_counts = malloc(actual_batches * sizeof(int));
    TokenizeBatchArgs **args_list = malloc(actual_batches * sizeof(TokenizeBatchArgs *));
    BatchPool **batch_pools = calloc(actual_batches, sizeof(BatchPool*));
    if (!thread_list || !batch_results || !batch_token_counts || !args_list) {
        free(thread_list); free(batch_results); free(batch_token_counts); free(args_list);
        free_string_array(words_split);
        return NULL;
    }
    memset(batch_token_counts, 0, actual_batches * sizeof(int)); 
    #ifdef TESTING
    printf("Starting batch tokenization with %d threads.\n", actual_batches);
    #endif
    for (int i = 0; i < actual_batches; i++) { 
        int start_index = i * batch_size; 
        int end_index = (i + 1) * batch_size < num_words ? (i + 1) * batch_size : num_words; 
        TokenizeBatchArgs *args = malloc(sizeof(TokenizeBatchArgs)); 
        args_list[i] = args;
        if (!args) {
            perror("Failed to allocate TokenizeBatchArgs");
            // cleanup
            for (int k = 0; k < i; k++) free(args_list[k]);
            free(args_list); free(batch_results); free(batch_token_counts); free(thread_list);
            free_string_array(words_split);
            return NULL;
        }
        args->tokenizer = tokenizer; 
        args->words_slice = &words_split[start_index]; 
        args->num_words_in_batch = end_index - start_index; 
        args->batch_results = &batch_results[i]; 
        args->batch_index = i; 
        args->main_out_token_count = &current_token_count; 
        args->batch_token_counts = &batch_token_counts; 
        args->target_core = i % NUM_CORES; // Distribute threads across cores

        /* allocate pool for this batch */
        int max_tokens_for_batch = 0;
        size_t total_chars = 0;
        int max_word_len = 0;
        for (int k = start_index; k < end_index; k++) { int l = strlen(words_split[k]); max_tokens_for_batch += l; total_chars += (size_t)l + 1; if (l > max_word_len) max_word_len = l; }
        BatchPool *pool = malloc(sizeof(BatchPool));
        if (pool) {
            pool->max_tokens = max_tokens_for_batch;
            pool->token_structs = malloc((size_t)max_tokens_for_batch * sizeof(tokenID));
            pool->arena_size = total_chars + (size_t)max_tokens_for_batch * 8 + 1024;
            pool->arena = malloc(pool->arena_size);
            pool->arena_off = 0;
            pool->malloced_strings = NULL;
            pool->malloced_count = 0;
            pool->malloced_cap = 0;
            /* allocate reusable scratch buffers sized to the longest word in this batch */
            pool->symbols_buf = NULL; pool->pair_buf = NULL; pool->symbols_cap = 0; pool->pair_cap = 0;
            if (max_word_len > 0) {
                pool->symbols_buf = malloc((size_t)max_word_len * sizeof(Symbol));
                pool->symbols_cap = pool->symbols_buf ? max_word_len : 0;
                pool->pair_buf = malloc((size_t)(max_word_len) * sizeof(Pair));
                pool->pair_cap = pool->pair_buf ? max_word_len : 0;
            }
            if (!pool->token_structs || !pool->arena) { if (pool->token_structs) free(pool->token_structs); if (pool->arena) free(pool->arena); free(pool); pool = NULL; }
        }
        batch_pools[i] = pool;
        args->pool = pool;

        if (args->num_words_in_batch <= 0) {
            batch_results[i] = NULL;
            batch_token_counts[i] = 0;
            free(args);
            args_list[i] = NULL;
            continue;
        }

        if (pthread_create(&thread_list[i], NULL, tokenizeBatchWrapper, args) != 0) {
            perror("Failed to create thread");
            free(args);
            // cleanup
            for (int k = 0; k < i; k++) if (args_list[k]) free(args_list[k]);
            free(args_list); free(batch_results); free(batch_token_counts); free(thread_list);
            free_string_array(words_split);
            return NULL;
        }

        #ifdef TESTING 
            printf("Thread for batch %d out of %d started on core %d.\n", i, actual_batches, args->target_core); 
        #endif 
    } 

    for (int i = 0; i < actual_batches; i++) { 
        if (args_list[i]) {
            pthread_join(thread_list[i], NULL);
            free(args_list[i]); // Free the arguments struct for this thread after it has completed
        }
        #ifdef TESTING 
            printf("Thread for batch %d out of %d merged.\n", i, actual_batches); 
        #endif 
    }
    free(args_list);
    #ifdef TESTING
     printf("All threads completed.\n");
    #endif
    /* compute total tokens from batch counts (threads no longer increment global counter) */
    current_token_count = 0;
    for (int i = 0; i < actual_batches; i++) current_token_count += batch_token_counts[i];
    #ifdef TESTING
     printf("Total tokens computed: %d\n", current_token_count);
    #endif
    free(thread_list);
    #ifdef TESTING
     printf("Merging batch results into final token list...\n");
    #endif
    tokens_list = malloc((current_token_count + 1) * sizeof(tokenID *));
    #ifdef TESTING
     printf("Allocated memory for final tokens list.\n");
    #endif
    int current_pos = 0;

    for (int i = 0; i < actual_batches; i++) {
        #ifdef TESTING
        printf("Merging batch %d with %d tokens.\n", i, batch_token_counts[i]);
        #endif
        int count = batch_token_counts[i];
        for (int j = 0; j < count; j++) {
            tokens_list[current_pos++] = batch_results[i][j];
        }
    }
    #ifdef TESTING
     printf("Batch results merged. Total tokens in final list: %d\n", current_pos);
    #endif
    for (int i = 0; i < actual_batches; i++) {
        if (batch_results[i]) free(batch_results[i]); // Free each batch's tokenID pointer array (structs live in batch pools)
    }
    free(batch_results); // Free the top-level batch results array
    free(batch_token_counts); // Free the batch token counts array
    free_string_array(words_split);
    if (tokens_list) tokens_list[current_token_count] = NULL;

    /* Build wrapper so caller can free batch pools once done */
    typedef struct { int pool_count; BatchPool **pools; } TokenListWrapper;
    TokenListWrapper *wrapper = malloc(sizeof(TokenListWrapper));
    if (!wrapper) {
        /* cleanup pools */
        for (int i = 0; i < actual_batches; i++) {
            if (batch_pools[i]) { free(batch_pools[i]->token_structs); free(batch_pools[i]->arena); free(batch_pools[i]); }
        }
        free(batch_pools);
        *out_token_count = current_token_count;
        return tokens_list;
    }
    wrapper->pool_count = actual_batches;
    wrapper->pools = batch_pools;

    /* combined block: [wrapper][token pointers...][NULL] */
    void **combined = malloc((1 + current_token_count + 1) * sizeof(void*));
    if (!combined) {
        free(wrapper);
        for (int i = 0; i < actual_batches; i++) {
            if (batch_pools[i]) { free(batch_pools[i]->token_structs); free(batch_pools[i]->arena); free(batch_pools[i]); }
        }
        free(batch_pools);
        *out_token_count = current_token_count;
        return tokens_list;
    }
    combined[0] = (void*)(((uintptr_t)wrapper) | 1);
    for (int i = 0; i < current_token_count; i++) combined[i + 1] = tokens_list[i];
    combined[current_token_count + 1] = NULL;
    free(tokens_list);
    *out_token_count = current_token_count;
    return (tokenID **)&combined[1];
}
char **decode(BPEtokenizer *tokenizer, tokenID **tokens, int token_count)
{
    if (!tokens || token_count <= 0)
        return NULL;
    char **out = malloc((token_count + 1) * sizeof(char *));
    if (!out)
        return NULL;
    for (int i = 0; i < token_count; i++)
    {
        char *token_str = find_token_by_id(tokenizer, tokens[i]->id);
        if (token_str)
        {
            out[i] = strdup(token_str);
        }
        else
        {
            out[i] = strdup(""); // Unknown token, to be replaced with the unknown token from the vocab if available, if not, then throw an error like modern tokenizers do
        }
    }
    out[token_count] = NULL;
    return out;
}
