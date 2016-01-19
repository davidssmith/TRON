#ifndef _RA_H
#define _RA_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <err.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>


/*
   File layout

   Additional info can be stored after the data region with no harmful effects.
*/
typedef struct {
    uint64_t flags;    /* file properties, such as endianness and future capabilities */
    uint64_t eltype;   /* enum representing the element type in the array */
    uint64_t elbyte;   /* # of bytes in type's canonical representation */
    uint64_t size;     /* size of data in bytes (may be compressed: check 'flags') */
    uint64_t ndims;    /* number of dimensions in array */
    uint64_t *dims;    /* the actual dimensions */
    uint8_t *data;     /* pointer to raw data -- contiguous -- so can mmap if y'ant'ta.
                          Use chars to handle generic data, since reader can use 'type'
                          enum to recreate correct pointer cast */
} ra_t;


static uint64_t RA_MAGIC_NUMBER = 0x7961727261776172ULL;

/* flags */
#define RA_FLAG_BIG_ENDIAN  (1ULL<<0)

/* maximum size that read system call can handle */
#define RA_MAX_BYTES  (1ULL<<31)

/* elemental types */
typedef enum {
    RA_TYPE_USER = 0, /* composite type, with optional elemental size
                          given by elbyte. User must handle decoding.
                          Note ras are recursive: a ra can contain
                          another ra */
    RA_TYPE_INT,
    RA_TYPE_UINT,
    RA_TYPE_FLOAT,
    RA_TYPE_COMPLEX
} ra_type;

static const char *RA_TYPE_NAMES[] = {
    "user",
    "int",
    "uint",
    "float",
    "complex" };

#ifdef __cplusplus
extern "C" {
#endif

int ra_read  (ra_t *a, const char *path);
void ra_query  (const char *path);
//uint64_t ra_data_offset (const char *path);   /* for mmap purposes */
//uint8_t ra_ndims (const char *path);
//uint8_t ra_type (const char *path);
int ra_write (ra_t *a, const char *path);
void ra_free (ra_t *a);

#ifdef __cplusplus
}
#endif

#endif   /* _ra_H */
