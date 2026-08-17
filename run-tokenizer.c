#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tokenizer BPE.h"

int main() {
    const char *filename = "large_text2.txt";


    // 2. Initialize Tokenizer
    BPEtokenizer t = initTokenizer("./wikitext103-bpe-tokenizer.json");
    if (t.vocab_size <= 0) {
        printf("Tokenizer initialization failed. Check file path.\n");
        return 1;
    }

    
    FILE *f = fopen(filename, "rb");
    if (f){
      //  printf("File '%s' opened successfully.\n", filename);
    }
    else {
        printf("Downloading 2MB of text...\n");
        system("powershell -Command \"Invoke-WebRequest 'https://raw.githubusercontent.com/Radoq10288/text-to-html/refs/heads/main/sample-2mb-text-file.txt' -OutFile 'large_text2.txt'\"");
        f = fopen(filename, "rb");
        if (!f) {
            printf("Failed to open the file after download attempt.\n");
            freeTokenizer(&t);
            return 1;
        }
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    // Limit to 2MB (2,097,152 bytes)
    if (fsize > 2*1048576) fsize = 2*1048576; 

    char *input_text = malloc(fsize + 1);
    if (!input_text) {
        printf("Memory allocation failed.\n");
        fclose(f);
        return 1;
    }

    fread(input_text, 1, fsize, f);
    fclose(f);
    input_text[fsize] = '\0'; // Ensure null-termination

    // 4. Tokenize
    int token_count = 0;
   // printf("Tokenizing %ld bytes from %s...\n", fsize, filename);
    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);
    tokenID **tokens = tokenize(&t, input_text, &token_count);
    QueryPerformanceCounter(&end);
    double elapsed_ms = (double)(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
    printf("tokenize() took %.3f ms\n", elapsed_ms);
    // 5. Cleanup
    free(input_text);
    freeTokenizer(&t);
    return 0;
}
