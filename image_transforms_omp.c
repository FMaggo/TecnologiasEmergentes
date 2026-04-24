#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

/* ─────────────────────────────────────────────
   Estructuras y utilidades BMP
   ───────────────────────────────────────────── */

#define HEADER_SIZE 54
#define BLUR_KERNEL  5   /* Tamaño del kernel de desenfoque (impar) */

typedef struct {
    unsigned char header[HEADER_SIZE];
    long width;
    long height;
    unsigned char *pixels; /* BGR interleaved, row-major */
} BmpImage;

/* Lee una imagen BMP de 24 bits. Devuelve 0 en error. */
int bmp_read(const char *filename, BmpImage *img) {
    FILE *f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "No se pudo abrir %s\n", filename); return 0; }

    if (fread(img->header, 1, HEADER_SIZE, f) != HEADER_SIZE) {
        fclose(f); return 0;
    }

    /* Ancho y alto desde cabecera (little-endian, offsets 18 y 22) */
    img->width  = img->header[18] | ((long)img->header[19] << 8) |
                  ((long)img->header[20] << 16) | ((long)img->header[21] << 24);
    img->height = img->header[22] | ((long)img->header[23] << 8) |
                  ((long)img->header[24] << 16) | ((long)img->header[25] << 24);

    long total = img->width * img->height * 3;
    img->pixels = (unsigned char *)malloc(total);
    if (!img->pixels) { fclose(f); return 0; }

    if ((long)fread(img->pixels, 1, total, f) != total) {
        /* Algunos BMP tienen padding de fila; intentamos leer píxel a píxel */
        rewind(f);
        fseek(f, HEADER_SIZE, SEEK_SET);
        long idx = 0;
        while (idx < total && !feof(f)) {
            img->pixels[idx++] = (unsigned char)fgetc(f);
        }
    }
    fclose(f);
    return 1;
}

/* Escribe imagen BMP de 24 bits. */
int bmp_write(const char *filename, const BmpImage *img) {
    FILE *f = fopen(filename, "wb");
    if (!f) { fprintf(stderr, "No se pudo crear %s\n", filename); return 0; }
    fwrite(img->header, 1, HEADER_SIZE, f);
    fwrite(img->pixels, 1, img->width * img->height * 3, f);
    fclose(f);
    return 1;
}

/* Libera memoria de píxeles. */
void bmp_free(BmpImage *img) {
    if (img->pixels) { free(img->pixels); img->pixels = NULL; }
}

/* ─────────────────────────────────────────────
   Transformaciones
   ───────────────────────────────────────────── */

/*
 * Convierte píxeles a escala de grises (luma BT.601).
 * Devuelve arreglo de bytes (un canal por píxel).
 */
unsigned char *to_grayscale(const BmpImage *src) {
    long n = src->width * src->height;
    unsigned char *gray = (unsigned char *)malloc(n);
    if (!gray) return NULL;

    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; i++) {
        unsigned char b = src->pixels[i * 3];
        unsigned char g = src->pixels[i * 3 + 1];
        unsigned char r = src->pixels[i * 3 + 2];
        gray[i] = (unsigned char)(0.07 * b + 0.72 * g + 0.21 * r);
    }
    return gray;
}

/* Construye BmpImage con canal gris replicado a BGR */
BmpImage gray_to_bmp(const BmpImage *src, const unsigned char *gray) {
    BmpImage dst;
    memcpy(dst.header, src->header, HEADER_SIZE);
    dst.width  = src->width;
    dst.height = src->height;
    long n = dst.width * dst.height;
    dst.pixels = (unsigned char *)malloc(n * 3);

    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; i++) {
        dst.pixels[i * 3]     = gray[i];
        dst.pixels[i * 3 + 1] = gray[i];
        dst.pixels[i * 3 + 2] = gray[i];
    }
    return dst;
}

/* 1. Inversión horizontal en escala de grises */
BmpImage transform_flip_h_gray(const BmpImage *src) {
    unsigned char *gray = to_grayscale(src);
    long W = src->width, H = src->height;
    unsigned char *flipped = (unsigned char *)malloc(W * H);

    #pragma omp parallel for schedule(static)
    for (long row = 0; row < H; row++) {
        for (long col = 0; col < W; col++) {
            flipped[row * W + col] = gray[row * W + (W - 1 - col)];
        }
    }
    free(gray);
    BmpImage result = gray_to_bmp(src, flipped);
    free(flipped);
    return result;
}

/* 2. Inversión vertical en escala de grises */
BmpImage transform_flip_v_gray(const BmpImage *src) {
    unsigned char *gray = to_grayscale(src);
    long W = src->width, H = src->height;
    unsigned char *flipped = (unsigned char *)malloc(W * H);

    #pragma omp parallel for schedule(static)
    for (long row = 0; row < H; row++) {
        memcpy(&flipped[row * W], &gray[(H - 1 - row) * W], W);
    }
    free(gray);
    BmpImage result = gray_to_bmp(src, flipped);
    free(flipped);
    return result;
}

/* 3. Desenfoque (box blur) en escala de grises */
BmpImage transform_blur_gray(const BmpImage *src) {
    unsigned char *gray = to_grayscale(src);
    long W = src->width, H = src->height;
    unsigned char *blurred = (unsigned char *)malloc(W * H);
    int half = BLUR_KERNEL / 2;

    #pragma omp parallel for schedule(static)
    for (long row = 0; row < H; row++) {
        for (long col = 0; col < W; col++) {
            long sum = 0, count = 0;
            for (int dy = -half; dy <= half; dy++) {
                for (int dx = -half; dx <= half; dx++) {
                    long r = row + dy, c = col + dx;
                    if (r >= 0 && r < H && c >= 0 && c < W) {
                        sum += gray[r * W + c];
                        count++;
                    }
                }
            }
            blurred[row * W + col] = (unsigned char)(sum / count);
        }
    }
    free(gray);
    BmpImage result = gray_to_bmp(src, blurred);
    free(blurred);
    return result;
}

/* 4. Inversión horizontal a color */
BmpImage transform_flip_h_color(const BmpImage *src) {
    long W = src->width, H = src->height;
    BmpImage dst;
    memcpy(dst.header, src->header, HEADER_SIZE);
    dst.width  = W; dst.height = H;
    dst.pixels = (unsigned char *)malloc(W * H * 3);

    #pragma omp parallel for schedule(static)
    for (long row = 0; row < H; row++) {
        for (long col = 0; col < W; col++) {
            long src_idx = (row * W + (W - 1 - col)) * 3;
            long dst_idx = (row * W + col) * 3;
            dst.pixels[dst_idx]     = src->pixels[src_idx];
            dst.pixels[dst_idx + 1] = src->pixels[src_idx + 1];
            dst.pixels[dst_idx + 2] = src->pixels[src_idx + 2];
        }
    }
    return dst;
}

/* 5. Inversión vertical a color */
BmpImage transform_flip_v_color(const BmpImage *src) {
    long W = src->width, H = src->height;
    BmpImage dst;
    memcpy(dst.header, src->header, HEADER_SIZE);
    dst.width  = W; dst.height = H;
    dst.pixels = (unsigned char *)malloc(W * H * 3);

    #pragma omp parallel for schedule(static)
    for (long row = 0; row < H; row++) {
        long src_row = H - 1 - row;
        memcpy(&dst.pixels[row * W * 3], &src->pixels[src_row * W * 3], W * 3);
    }
    return dst;
}

/* 6. Desenfoque (box blur) a color */
BmpImage transform_blur_color(const BmpImage *src) {
    long W = src->width, H = src->height;
    BmpImage dst;
    memcpy(dst.header, src->header, HEADER_SIZE);
    dst.width  = W; dst.height = H;
    dst.pixels = (unsigned char *)malloc(W * H * 3);
    int half = BLUR_KERNEL / 2;

    #pragma omp parallel for schedule(static)
    for (long row = 0; row < H; row++) {
        for (long col = 0; col < W; col++) {
            long sumB = 0, sumG = 0, sumR = 0, count = 0;
            for (int dy = -half; dy <= half; dy++) {
                for (int dx = -half; dx <= half; dx++) {
                    long r = row + dy, c = col + dx;
                    if (r >= 0 && r < H && c >= 0 && c < W) {
                        long idx = (r * W + c) * 3;
                        sumB += src->pixels[idx];
                        sumG += src->pixels[idx + 1];
                        sumR += src->pixels[idx + 2];
                        count++;
                    }
                }
            }
            long out = (row * W + col) * 3;
            dst.pixels[out]     = (unsigned char)(sumB / count);
            dst.pixels[out + 1] = (unsigned char)(sumG / count);
            dst.pixels[out + 2] = (unsigned char)(sumR / count);
        }
    }
    return dst;
}

/* ─────────────────────────────────────────────
   Función: procesar una imagen con las 6 tareas
   ───────────────────────────────────────────── */

void process_image(const char *input_path, int num_threads) {
    BmpImage src;
    if (!bmp_read(input_path, &src)) {
        fprintf(stderr, "[ERROR] No se pudo leer: %s\n", input_path);
        return;
    }

    printf("\n[IMG] %s  (%ld x %ld px)\n", input_path, src.width, src.height);
    printf("[OMP] Threads: %d\n", num_threads);

    /* Extraer nombre base para archivos de salida */
    char base[256];
    strncpy(base, input_path, sizeof(base) - 1);
    /* Quitar extensión */
    char *dot = strrchr(base, '.');
    if (dot) *dot = '\0';

    /* Construir nombres de salida */
    char out[6][300];
    snprintf(out[0], sizeof(out[0]), "%s_flip_h_gray.bmp",  base);
    snprintf(out[1], sizeof(out[1]), "%s_flip_v_gray.bmp",  base);
    snprintf(out[2], sizeof(out[2]), "%s_blur_gray.bmp",    base);
    snprintf(out[3], sizeof(out[3]), "%s_flip_h_color.bmp", base);
    snprintf(out[4], sizeof(out[4]), "%s_flip_v_color.bmp", base);
    snprintf(out[5], sizeof(out[5]), "%s_blur_color.bmp",   base);

    omp_set_num_threads(num_threads);

    double t_start = omp_get_wtime();

    /* ── PARALELISMO A NIVEL DE TAREAS ── */
    #pragma omp parallel
    {
        #pragma omp single
        {
            /* Tarea 1: Flip horizontal gris */
            #pragma omp task
            {
                double t0 = omp_get_wtime();
                BmpImage r = transform_flip_h_gray(&src);
                bmp_write(out[0], &r); bmp_free(&r);
                printf("  [T%d] flip_h_gray    -> %.3f s\n",
                       omp_get_thread_num(), omp_get_wtime() - t0);
            }

            /* Tarea 2: Flip vertical gris */
            #pragma omp task
            {
                double t0 = omp_get_wtime();
                BmpImage r = transform_flip_v_gray(&src);
                bmp_write(out[1], &r); bmp_free(&r);
                printf("  [T%d] flip_v_gray    -> %.3f s\n",
                       omp_get_thread_num(), omp_get_wtime() - t0);
            }

            /* Tarea 3: Blur gris */
            #pragma omp task
            {
                double t0 = omp_get_wtime();
                BmpImage r = transform_blur_gray(&src);
                bmp_write(out[2], &r); bmp_free(&r);
                printf("  [T%d] blur_gray      -> %.3f s\n",
                       omp_get_thread_num(), omp_get_wtime() - t0);
            }

            /* Tarea 4: Flip horizontal color */
            #pragma omp task
            {
                double t0 = omp_get_wtime();
                BmpImage r = transform_flip_h_color(&src);
                bmp_write(out[3], &r); bmp_free(&r);
                printf("  [T%d] flip_h_color   -> %.3f s\n",
                       omp_get_thread_num(), omp_get_wtime() - t0);
            }

            /* Tarea 5: Flip vertical color */
            #pragma omp task
            {
                double t0 = omp_get_wtime();
                BmpImage r = transform_flip_v_color(&src);
                bmp_write(out[4], &r); bmp_free(&r);
                printf("  [T%d] flip_v_color   -> %.3f s\n",
                       omp_get_thread_num(), omp_get_wtime() - t0);
            }

            /* Tarea 6: Blur color */
            #pragma omp task
            {
                double t0 = omp_get_wtime();
                BmpImage r = transform_blur_color(&src);
                bmp_write(out[5], &r); bmp_free(&r);
                printf("  [T%d] blur_color     -> %.3f s\n",
                       omp_get_thread_num(), omp_get_wtime() - t0);
            }

            #pragma omp taskwait
        } /* end single */
    } /* end parallel */

    double elapsed = omp_get_wtime() - t_start;
    printf("[TOTAL] %s  con %d threads -> %.3f s\n",
           input_path, num_threads, elapsed);

    bmp_free(&src);
}

/* ─────────────────────────────────────────────
   main
   ───────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    if (argc < 5) {
        fprintf(stderr,
            "Uso: %s <img1.bmp> <img2.bmp> <img3.bmp> <num_threads>\n",
            argv[0]);
        fprintf(stderr,
            "Ejemplo: %s foto1.bmp foto2.bmp foto3.bmp 12\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[4]);
    if (num_threads < 1) { fprintf(stderr, "num_threads debe ser >= 1\n"); return 1; }

    printf("=============================================\n");
    printf(" Transformaciones BMP con OpenMP (tasks)\n");
    printf(" Kernel de blur: %dx%d\n", BLUR_KERNEL, BLUR_KERNEL);
    printf(" Threads solicitados: %d\n", num_threads);
    printf("=============================================\n");

    /* Procesar las tres imágenes secuencialmente (cada una distribuye sus 6
       tareas en paralelo sobre los threads disponibles) */
    for (int i = 1; i <= 3; i++) {
        process_image(argv[i], num_threads);
    }

    printf("\n[DONE] Todas las transformaciones completadas.\n");
    return 0;
}
