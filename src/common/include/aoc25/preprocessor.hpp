#pragma once

// Macro to perform token pasting (no expansion).
#define CONCATENATE_NX(A, B) A##B

// Macro to force expansion of arguments (the indirection).
#define CONCATENATE(A, B) CONCATENATE_NX(A, B)

// Macro to stringify its argument (adding double quotes).
#define STRINGIFY_NX(x) #x

// Macro to force expansion before stringifying.
#define STRINGIFY(x) STRINGIFY_NX(x)
