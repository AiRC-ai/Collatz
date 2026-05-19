#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BIG_BASE 1000000000U
#define BIG_BASE_DIGITS 9

typedef struct {
    uint32_t *digits;
    size_t len;
    size_t cap;
} BigInt;

typedef struct {
    double *values;
    size_t len;
    size_t cap;
} DoubleVec;

typedef struct {
    size_t width;
    size_t height;
    size_t max_steps;
    size_t sequence_limit;
    int print_sequence;
    int print_all_sequence;
    const char *csv_path;
    const char *svg_path;
    const char *number;
} Options;

static void die(const char *message) {
    fprintf(stderr, "error: %s\n", message);
    exit(1);
}

static void *xmalloc(size_t size) {
    void *ptr = malloc(size);
    if (!ptr) {
        die("out of memory");
    }
    return ptr;
}

static void *xrealloc(void *ptr, size_t size) {
    void *next = realloc(ptr, size);
    if (!next) {
        die("out of memory");
    }
    return next;
}

static void big_init(BigInt *n) {
    n->cap = 4;
    n->len = 1;
    n->digits = xmalloc(n->cap * sizeof(*n->digits));
    n->digits[0] = 0;
}

static void big_free(BigInt *n) {
    free(n->digits);
    n->digits = NULL;
    n->len = 0;
    n->cap = 0;
}

static void big_reserve(BigInt *n, size_t need) {
    if (need <= n->cap) {
        return;
    }
    while (n->cap < need) {
        n->cap *= 2;
    }
    n->digits = xrealloc(n->digits, n->cap * sizeof(*n->digits));
}

static void big_trim(BigInt *n) {
    while (n->len > 1 && n->digits[n->len - 1] == 0) {
        n->len--;
    }
}

static int big_is_zero(const BigInt *n) {
    return n->len == 1 && n->digits[0] == 0;
}

static int big_is_one(const BigInt *n) {
    return n->len == 1 && n->digits[0] == 1;
}

static int big_is_even(const BigInt *n) {
    return (n->digits[0] % 2U) == 0U;
}

static void big_mul_small(BigInt *n, uint32_t m) {
    uint64_t carry = 0;
    for (size_t i = 0; i < n->len; i++) {
        uint64_t cur = (uint64_t)n->digits[i] * m + carry;
        n->digits[i] = (uint32_t)(cur % BIG_BASE);
        carry = cur / BIG_BASE;
    }
    while (carry > 0) {
        big_reserve(n, n->len + 1);
        n->digits[n->len++] = (uint32_t)(carry % BIG_BASE);
        carry /= BIG_BASE;
    }
}

static void big_add_small(BigInt *n, uint32_t add) {
    uint64_t carry = add;
    size_t i = 0;
    while (carry > 0) {
        if (i == n->len) {
            big_reserve(n, n->len + 1);
            n->digits[n->len++] = 0;
        }
        uint64_t cur = (uint64_t)n->digits[i] + carry;
        n->digits[i] = (uint32_t)(cur % BIG_BASE);
        carry = cur / BIG_BASE;
        i++;
    }
}

static void big_from_decimal(BigInt *n, const char *text) {
    big_init(n);

    while (isspace((unsigned char)*text)) {
        text++;
    }
    if (*text == '+') {
        text++;
    }
    if (!isdigit((unsigned char)*text)) {
        die("number must be a positive whole number");
    }

    for (; *text; text++) {
        if (isspace((unsigned char)*text)) {
            while (isspace((unsigned char)*text)) {
                text++;
            }
            if (*text != '\0') {
                die("number cannot contain embedded whitespace");
            }
            break;
        }
        if (!isdigit((unsigned char)*text)) {
            die("number must contain only decimal digits");
        }
        big_mul_small(n, 10);
        big_add_small(n, (uint32_t)(*text - '0'));
    }

    big_trim(n);
    if (big_is_zero(n)) {
        die("Collatz input must be greater than zero");
    }
}

static void big_div2(BigInt *n) {
    uint64_t carry = 0;
    for (size_t i = n->len; i-- > 0;) {
        uint64_t cur = n->digits[i] + carry * BIG_BASE;
        n->digits[i] = (uint32_t)(cur / 2U);
        carry = cur % 2U;
    }
    big_trim(n);
}

static void big_mul3_add1(BigInt *n) {
    uint64_t carry = 1;
    for (size_t i = 0; i < n->len; i++) {
        uint64_t cur = (uint64_t)n->digits[i] * 3U + carry;
        n->digits[i] = (uint32_t)(cur % BIG_BASE);
        carry = cur / BIG_BASE;
    }
    while (carry > 0) {
        big_reserve(n, n->len + 1);
        n->digits[n->len++] = (uint32_t)(carry % BIG_BASE);
        carry /= BIG_BASE;
    }
}

static char *big_to_string(const BigInt *n) {
    size_t cap = n->len * BIG_BASE_DIGITS + 2;
    char *out = xmalloc(cap);
    char *p = out;
    int written = snprintf(p, cap, "%u", n->digits[n->len - 1]);
    if (written < 0) {
        die("failed to format number");
    }
    p += written;
    cap -= (size_t)written;

    for (size_t i = n->len - 1; i-- > 0;) {
        written = snprintf(p, cap, "%09u", n->digits[i]);
        if (written < 0) {
            die("failed to format number");
        }
        p += written;
        cap -= (size_t)written;
    }

    return out;
}

static double big_log10(const BigInt *n) {
    if (big_is_zero(n)) {
        return 0.0;
    }
    return log10((double)n->digits[n->len - 1]) +
           (double)(n->len - 1) * BIG_BASE_DIGITS;
}

static void vec_push(DoubleVec *vec, double value) {
    if (vec->len == vec->cap) {
        vec->cap = vec->cap == 0 ? 256 : vec->cap * 2;
        vec->values = xrealloc(vec->values, vec->cap * sizeof(*vec->values));
    }
    vec->values[vec->len++] = value;
}

static void vec_free(DoubleVec *vec) {
    free(vec->values);
    vec->values = NULL;
    vec->len = 0;
    vec->cap = 0;
}

static size_t parse_size(const char *text, const char *name, size_t min, size_t max) {
    char *end = NULL;
    unsigned long long value;

    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno || end == text || *end != '\0') {
        fprintf(stderr, "error: invalid %s: %s\n", name, text);
        exit(1);
    }
    if (value < min || value > max) {
        fprintf(stderr, "error: %s must be between %zu and %zu\n", name, min, max);
        exit(1);
    }
    return (size_t)value;
}

static void usage(const char *program) {
    printf("Usage: %s NUMBER [options]\n", program);
    printf("\n");
    printf("Runs the 3n + 1 / Collatz sequence and draws a terminal plot.\n");
    printf("\n");
    printf("Options:\n");
    printf("  --width N           Plot width, default 100\n");
    printf("  --height N          Plot height, default 24\n");
    printf("  --max-steps N       Stop after N steps if 1 is not reached, default 1000000\n");
    printf("  --sequence-limit N  Print at most N sequence rows before capping, default 200\n");
    printf("  --all-sequence      Print every sequence row\n");
    printf("  --no-sequence       Only print summary and plot\n");
    printf("  --csv FILE          Save step,value,log10_value data for outside plotting\n");
    printf("  --svg FILE          Save a browser-friendly SVG plot\n");
    printf("  --help              Show this help\n");
}

static Options parse_options(int argc, char **argv) {
    Options opt = {
        .width = 100,
        .height = 24,
        .max_steps = 1000000,
        .sequence_limit = 200,
        .print_sequence = 1,
        .print_all_sequence = 0,
        .csv_path = NULL,
        .svg_path = NULL,
        .number = NULL,
    };

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            exit(0);
        } else if (strcmp(argv[i], "--width") == 0) {
            if (++i >= argc) {
                die("--width requires a value");
            }
            opt.width = parse_size(argv[i], "width", 20, 240);
        } else if (strcmp(argv[i], "--height") == 0) {
            if (++i >= argc) {
                die("--height requires a value");
            }
            opt.height = parse_size(argv[i], "height", 8, 80);
        } else if (strcmp(argv[i], "--max-steps") == 0) {
            if (++i >= argc) {
                die("--max-steps requires a value");
            }
            opt.max_steps = parse_size(argv[i], "max steps", 1, 1000000000);
        } else if (strcmp(argv[i], "--sequence-limit") == 0) {
            if (++i >= argc) {
                die("--sequence-limit requires a value");
            }
            opt.sequence_limit = parse_size(argv[i], "sequence limit", 0, 1000000000);
        } else if (strcmp(argv[i], "--all-sequence") == 0) {
            opt.print_all_sequence = 1;
            opt.print_sequence = 1;
        } else if (strcmp(argv[i], "--no-sequence") == 0) {
            opt.print_sequence = 0;
        } else if (strcmp(argv[i], "--csv") == 0) {
            if (++i >= argc) {
                die("--csv requires a file path");
            }
            opt.csv_path = argv[i];
        } else if (strcmp(argv[i], "--svg") == 0) {
            if (++i >= argc) {
                die("--svg requires a file path");
            }
            opt.svg_path = argv[i];
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "error: unknown option: %s\n", argv[i]);
            exit(1);
        } else if (!opt.number) {
            opt.number = argv[i];
        } else {
            die("only one input number is allowed");
        }
    }

    if (!opt.number) {
        usage(argv[0]);
        exit(1);
    }

    return opt;
}

static void maybe_print_sequence(const Options *opt, size_t step, const BigInt *n) {
    if (!opt->print_sequence) {
        return;
    }

    if (opt->print_all_sequence || step < opt->sequence_limit || big_is_one(n)) {
        char *text = big_to_string(n);
        printf("%8zu  %s\n", step, text);
        free(text);
    } else if (step == opt->sequence_limit) {
        printf("     ...  sequence output capped at %zu rows; use --all-sequence to print all\n",
               opt->sequence_limit);
    }
}

static void write_csv_row(FILE *csv, size_t step, const BigInt *n, double log_value) {
    if (!csv) {
        return;
    }

    char *text = big_to_string(n);
    fprintf(csv, "%zu,%s,%.17g\n", step, text, log_value);
    free(text);
}

static void draw_plot(const DoubleVec *points, size_t width, size_t height) {
    double min = 0.0;
    double max = 0.0;
    char *grid;

    for (size_t i = 0; i < points->len; i++) {
        if (points->values[i] > max) {
            max = points->values[i];
        }
    }
    if (max <= min) {
        max = 1.0;
    }

    grid = xmalloc(width * height);
    memset(grid, ' ', width * height);

    for (size_t r = 0; r < height; r++) {
        grid[r * width] = '|';
    }
    for (size_t c = 0; c < width; c++) {
        grid[(height - 1) * width + c] = '-';
    }
    grid[(height - 1) * width] = '+';

    for (size_t i = 0; i < points->len; i++) {
        double x_norm = points->len == 1 ? 0.0 : (double)i / (double)(points->len - 1);
        double y_norm = (points->values[i] - min) / (max - min);
        size_t col = (size_t)llround(x_norm * (double)(width - 1));
        size_t row = height - 1 - (size_t)llround(y_norm * (double)(height - 1));
        char *cell = &grid[row * width + col];
        *cell = (*cell == ' ' || *cell == '|' || *cell == '-') ? '*' : '#';
    }

    printf("\nPlot map: x = step, y = log10(value)\n\n");
    for (size_t r = 0; r < height; r++) {
        double label = max - ((double)r / (double)(height - 1)) * (max - min);
        printf("%7.2f ", label);
        fwrite(&grid[r * width], 1, width, stdout);
        putchar('\n');
    }
    printf("        step 0");
    if (width > 24) {
        for (size_t i = 0; i < width - 24; i++) {
            putchar(' ');
        }
    } else {
        putchar(' ');
    }
    printf("step %zu\n", points->len - 1);

    free(grid);
}

static void write_svg_plot(const DoubleVec *points, const char *path) {
    const int width = 1000;
    const int height = 600;
    const int left = 76;
    const int right = 28;
    const int top = 28;
    const int bottom = 62;
    const int plot_w = width - left - right;
    const int plot_h = height - top - bottom;
    double min = 0.0;
    double max = 0.0;
    FILE *svg = fopen(path, "w");

    if (!svg) {
        perror(path);
        exit(1);
    }

    for (size_t i = 0; i < points->len; i++) {
        if (points->values[i] > max) {
            max = points->values[i];
        }
    }
    if (max <= min) {
        max = 1.0;
    }

    fprintf(svg, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    fprintf(svg, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\">\n",
            width, height, width, height);
    fprintf(svg, "<rect width=\"100%%\" height=\"100%%\" fill=\"#05072a\"/>\n");
    fprintf(svg, "<text x=\"%d\" y=\"24\" fill=\"#ffffff\" font-family=\"Menlo, Consolas, monospace\" font-size=\"18\">3n + 1 Collatz plot</text>\n",
            left);
    fprintf(svg, "<text x=\"%d\" y=\"%d\" fill=\"#b8c7ff\" font-family=\"Menlo, Consolas, monospace\" font-size=\"13\">x = step, y = log10(value)</text>\n",
            left, height - 18);

    for (int i = 0; i <= 5; i++) {
        double t = (double)i / 5.0;
        int y = top + (int)llround(t * plot_h);
        double label = max - t * (max - min);
        fprintf(svg, "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"#182052\" stroke-width=\"1\"/>\n",
                left, y, left + plot_w, y);
        fprintf(svg, "<text x=\"10\" y=\"%d\" fill=\"#b8c7ff\" font-family=\"Menlo, Consolas, monospace\" font-size=\"12\">%.2f</text>\n",
                y + 4, label);
    }

    fprintf(svg, "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"#d9e3ff\" stroke-width=\"2\"/>\n",
            left, top, left, top + plot_h);
    fprintf(svg, "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"#d9e3ff\" stroke-width=\"2\"/>\n",
            left, top + plot_h, left + plot_w, top + plot_h);

    fprintf(svg, "<text x=\"%d\" y=\"%d\" fill=\"#ffffff\" font-family=\"Menlo, Consolas, monospace\" font-size=\"12\">0</text>\n",
            left, top + plot_h + 22);
    fprintf(svg, "<text x=\"%d\" y=\"%d\" text-anchor=\"end\" fill=\"#ffffff\" font-family=\"Menlo, Consolas, monospace\" font-size=\"12\">%zu</text>\n",
            left + plot_w, top + plot_h + 22, points->len - 1);

    fprintf(svg, "<polyline fill=\"none\" stroke=\"#6ee7ff\" stroke-width=\"2.25\" stroke-linejoin=\"round\" stroke-linecap=\"round\" points=\"");
    for (size_t i = 0; i < points->len; i++) {
        double x_norm = points->len == 1 ? 0.0 : (double)i / (double)(points->len - 1);
        double y_norm = (points->values[i] - min) / (max - min);
        int x = left + (int)llround(x_norm * plot_w);
        int y = top + plot_h - (int)llround(y_norm * plot_h);
        fprintf(svg, "%d,%d ", x, y);
    }
    fprintf(svg, "\"/>\n");

    if (points->len <= 2000) {
        for (size_t i = 0; i < points->len; i++) {
            double x_norm = points->len == 1 ? 0.0 : (double)i / (double)(points->len - 1);
            double y_norm = (points->values[i] - min) / (max - min);
            int x = left + (int)llround(x_norm * plot_w);
            int y = top + plot_h - (int)llround(y_norm * plot_h);
            fprintf(svg, "<circle cx=\"%d\" cy=\"%d\" r=\"2.25\" fill=\"#ffffff\" opacity=\"0.78\"/>\n", x, y);
        }
    }

    fprintf(svg, "</svg>\n");
    fclose(svg);
}

int main(int argc, char **argv) {
    Options opt = parse_options(argc, argv);
    BigInt n;
    DoubleVec points = {0};
    FILE *csv = NULL;
    char *peak = NULL;
    size_t peak_step = 0;
    double peak_log = 0.0;
    size_t step = 0;
    int reached_one = 0;

    big_from_decimal(&n, opt.number);
    peak = big_to_string(&n);
    peak_log = big_log10(&n);

    if (opt.csv_path) {
        csv = fopen(opt.csv_path, "w");
        if (!csv) {
            perror(opt.csv_path);
            big_free(&n);
            free(peak);
            return 1;
        }
        fprintf(csv, "step,value,log10_value\n");
    }

    if (opt.print_sequence) {
        printf("Collatz sequence\n");
        printf("    step  value\n");
    }

    for (;;) {
        double log_value = big_log10(&n);
        vec_push(&points, log_value);
        maybe_print_sequence(&opt, step, &n);
        write_csv_row(csv, step, &n, log_value);

        if (log_value > peak_log) {
            free(peak);
            peak = big_to_string(&n);
            peak_step = step;
            peak_log = log_value;
        }

        if (big_is_one(&n)) {
            reached_one = 1;
            break;
        }
        if (step >= opt.max_steps) {
            break;
        }

        if (big_is_even(&n)) {
            big_div2(&n);
        } else {
            big_mul3_add1(&n);
        }
        step++;
    }

    if (csv) {
        fclose(csv);
    }

    printf("\nSummary\n");
    printf("  start:      %s\n", opt.number);
    printf("  result:     %s\n", reached_one ? "reached 1" : "stopped before reaching 1");
    printf("  steps:      %zu\n", step);
    printf("  peak step:  %zu\n", peak_step);
    printf("  peak value: %s\n", peak);
    if (opt.csv_path) {
        printf("  csv:        %s\n", opt.csv_path);
    }
    if (opt.svg_path) {
        write_svg_plot(&points, opt.svg_path);
        printf("  svg:        %s\n", opt.svg_path);
    }

    draw_plot(&points, opt.width, opt.height);

    free(peak);
    vec_free(&points);
    big_free(&n);
    return reached_one ? 0 : 2;
}
