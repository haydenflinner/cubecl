// Scalar fallbacks for the `__std_*` vectorized-algorithm exports that MSVC's
// STL provides only in VS2022+ toolsets. The bundled tracel LLVM objects were
// built against a newer msvcprt; linking them with VS2019's libs leaves these
// unresolved. Signatures/semantics match microsoft/STL's vector_algorithms.cpp.
// These run inside LLVM's JIT passes only — scalar speed is fine.

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace {

template <class T> size_t count_eq(const void* f, const void* l, T v) {
    size_t n = 0;
    for (auto p = (const T*)f; p != (const T*)l; ++p)
        n += *p == v;
    return n;
}

template <class T> const void* find_first(const void* f, const void* l, T v) {
    for (auto p = (const T*)f; p != (const T*)l; ++p)
        if (*p == v) return p;
    return l;
}

// std::search: first occurrence of needle [f2, f2+c2) in haystack, else last1.
const uint8_t* search_impl(const uint8_t* f1, const uint8_t* l1, const uint8_t* f2, size_t c2) {
    const size_t n1 = (size_t)(l1 - f1);
    if (c2 == 0) return f1;
    for (size_t i = 0; i + c2 <= n1; ++i) {
        if (memcmp(f1 + i, f2, c2) == 0) return f1 + i;
    }
    return l1;
}

// std::mismatch: position of the first elementwise difference.
template <class T> size_t mismatch_impl(const void* f1, const void* f2, size_t n) {
    auto a = (const T*)f1;
    auto b = (const T*)f2;
    size_t i = 0;
    while (i != n && a[i] == b[i]) ++i;
    return i;
}

// std::remove (stable): compact non-matching elements; returns the new end.
template <class T> void* remove_impl(void* f, void* l, T v) {
    auto rd = (const T*)f;
    auto wr = (T*)f;
    for (; rd != (const T*)l; ++rd)
        if (*rd != v) *wr++ = *rd;
    return wr;
}

// std::replace (in place).
template <class T> void replace_impl(void* f, void* l, T o, T n) {
    for (auto p = (T*)f; p != (T*)l; ++p)
        if (*p == o) *p = n;
}

// std::min/std::max over a range: return the extreme VALUE.
template <class T> T min_val(const void* f, const void* l) {
    auto p = (const T*)f;
    T m = *p;
    for (++p; p != (const T*)l; ++p)
        if (*p < m) m = *p;
    return m;
}
template <class T> T max_val(const void* f, const void* l) {
    auto p = (const T*)f;
    T m = *p;
    for (++p; p != (const T*)l; ++p)
        if (m < *p) m = *p;
    return m;
}

// std::min_element/max_element: pointer to the FIRST smallest / largest element.
template <class S, class U> const void* min_el(const void* f, const void* l, bool sgn) {
    if (f == l) return f;
    auto p = (const uint8_t*)f;
    const uint8_t* r = p;
    for (p += sizeof(S); p != (const uint8_t*)l; p += sizeof(S)) {
        const bool less = sgn ? (*(const S*)p < *(const S*)r) : (*(const U*)p < *(const U*)r);
        if (less) r = p;
    }
    return r;
}
template <class S, class U> const void* max_el(const void* f, const void* l, bool sgn) {
    if (f == l) return f;
    auto p = (const uint8_t*)f;
    const uint8_t* r = p;
    for (p += sizeof(S); p != (const uint8_t*)l; p += sizeof(S)) {
        const bool less = sgn ? (*(const S*)p < *(const S*)r) : (*(const U*)p < *(const U*)r);
        if (less) r = p;
    }
    return r;
}

} // unnamed namespace

extern "C" __declspec(dllimport) void __stdcall Sleep(unsigned long ms);

extern "C" {

// `printf` has no DLL export on Windows (it is a header inline over
// `__stdio_common_vfprintf`); JIT'd kernels calling variadic `printf` get
// this forwarder instead. The caller-side varargs ABI is intact — only the
// callee needs to exist as a real symbol.
int __cdecl __cubecl_jit_printf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int r = vprintf(fmt, ap);
    va_end(ap);
    return r;
}

void __stdcall _Thrd_sleep_for(unsigned long ms) noexcept {
    Sleep(ms);
}

[[noreturn]] void __stdcall __std_init_once_link_alternate_names_and_abort() noexcept {
    abort();
}

__declspec(noalias) size_t __stdcall __std_count_trivial_1(const void* f, const void* l, uint8_t v) noexcept {
    return count_eq(f, l, v);
}
__declspec(noalias) size_t __stdcall __std_count_trivial_4(const void* f, const void* l, uint32_t v) noexcept {
    return count_eq(f, l, v);
}
__declspec(noalias) size_t __stdcall __std_count_trivial_8(const void* f, const void* l, uint64_t v) noexcept {
    return count_eq(f, l, v);
}

const void* __stdcall __std_find_trivial_1(const void* f, const void* l, uint8_t v) noexcept {
    return find_first(f, l, v);
}
const void* __stdcall __std_find_trivial_2(const void* f, const void* l, uint16_t v) noexcept {
    return find_first(f, l, v);
}
const void* __stdcall __std_find_trivial_4(const void* f, const void* l, uint32_t v) noexcept {
    return find_first(f, l, v);
}
const void* __stdcall __std_find_trivial_8(const void* f, const void* l, uint64_t v) noexcept {
    return find_first(f, l, v);
}

const void* __stdcall __std_find_last_trivial_1(const void* f, const void* l, uint8_t v) noexcept {
    for (auto p = (const uint8_t*)l; p != (const uint8_t*)f;)
        if (*--p == v) return p;
    return l;
}

const void* __stdcall __std_search_1(const void* f1, const void* l1, const void* f2, size_t c2) noexcept {
    return search_impl((const uint8_t*)f1, (const uint8_t*)l1, (const uint8_t*)f2, c2);
}

// std::find_end: last occurrence of the needle, else last1.
const void* __stdcall __std_find_end_1(const void* f1, const void* l1, const void* f2, size_t c2) noexcept {
    const uint8_t* f = (const uint8_t*)f1;
    const uint8_t* l = (const uint8_t*)l1;
    const size_t n1 = (size_t)(l - f);
    if (c2 == 0 || c2 > n1) return l1;
    for (size_t i = n1 - c2 + 1; i-- > 0;) {
        if (memcmp(f + i, f2, c2) == 0) return f + i;
    }
    return l1;
}

// std::find_first_of as an index: first haystack[i] that appears in the needle
// set, else haystack_length.
__declspec(noalias) size_t __stdcall __std_find_first_of_trivial_pos_1(
    const void* haystack, size_t haystack_length, const void* needle, size_t needle_length) noexcept {
    auto h = (const uint8_t*)haystack;
    auto n = (const uint8_t*)needle;
    for (size_t i = 0; i < haystack_length; ++i)
        for (size_t j = 0; j < needle_length; ++j)
            if (h[i] == n[j]) return i;
    return haystack_length;
}

__declspec(noalias) size_t __stdcall __std_mismatch_1(const void* f1, const void* f2, size_t n) noexcept {
    return mismatch_impl<uint8_t>(f1, f2, n);
}
__declspec(noalias) size_t __stdcall __std_mismatch_4(const void* f1, const void* f2, size_t n) noexcept {
    return mismatch_impl<uint32_t>(f1, f2, n);
}
__declspec(noalias) size_t __stdcall __std_mismatch_8(const void* f1, const void* f2, size_t n) noexcept {
    return mismatch_impl<uint64_t>(f1, f2, n);
}

void* __stdcall __std_remove_1(void* f, void* l, uint8_t v) noexcept {
    return remove_impl(f, l, v);
}
void* __stdcall __std_remove_8(void* f, void* l, uint64_t v) noexcept {
    return remove_impl(f, l, v);
}

__declspec(noalias) void __stdcall __std_replace_4(void* f, void* l, uint32_t o, uint32_t n) noexcept {
    replace_impl(f, l, o, n);
}
__declspec(noalias) void __stdcall __std_replace_8(void* f, void* l, uint64_t o, uint64_t n) noexcept {
    replace_impl(f, l, o, n);
}

__declspec(noalias) int32_t __stdcall __std_min_4i(const void* f, const void* l) noexcept {
    return min_val<int32_t>(f, l);
}
__declspec(noalias) uint32_t __stdcall __std_min_4u(const void* f, const void* l) noexcept {
    return min_val<uint32_t>(f, l);
}
__declspec(noalias) uint32_t __stdcall __std_max_4u(const void* f, const void* l) noexcept {
    return max_val<uint32_t>(f, l);
}
__declspec(noalias) uint64_t __stdcall __std_max_8u(const void* f, const void* l) noexcept {
    return max_val<uint64_t>(f, l);
}

// `_Signed` picks the comparison domain.
const void* __stdcall __std_min_element_4(const void* f, const void* l, bool sgn) noexcept {
    return min_el<int32_t, uint32_t>(f, l, sgn);
}
const void* __stdcall __std_max_element_1(const void* f, const void* l, bool sgn) noexcept {
    return max_el<int8_t, uint8_t>(f, l, sgn);
}
const void* __stdcall __std_max_element_4(const void* f, const void* l, bool sgn) noexcept {
    return max_el<int32_t, uint32_t>(f, l, sgn);
}
const void* __stdcall __std_max_element_8(const void* f, const void* l, bool sgn) noexcept {
    return max_el<int64_t, uint64_t>(f, l, sgn);
}

struct _Min_max_element_t {
    const void* _Min;
    const void* _Max;
};
// std::minmax_element: _Min = first smallest, _Max = LAST largest.
_Min_max_element_t __stdcall __std_minmax_element_8(const void* f, const void* l, bool sgn) noexcept {
    _Min_max_element_t r{f, f};
    if (f == l) return r;
    auto p = (const uint8_t*)f;
    const uint8_t* mn = p;
    const uint8_t* mx = p;
    for (p += 8; p != (const uint8_t*)l; p += 8) {
        if (sgn) {
            if (*(const int64_t*)p < *(const int64_t*)mn) mn = p;
            if (*(const int64_t*)p >= *(const int64_t*)mx) mx = p;
        } else {
            if (*(const uint64_t*)p < *(const uint64_t*)mn) mn = p;
            if (*(const uint64_t*)p >= *(const uint64_t*)mx) mx = p;
        }
    }
    r._Min = mn;
    r._Max = mx;
    return r;
}

} // extern "C"
