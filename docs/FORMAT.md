# The OpenZC container format, version 2.0

This is the normative description of the format. It is written so that a decoder
can be implemented from this document alone, without linking the `openzc` crate —
`crates/openzc/src/format/` is the reference implementation, and the two are
kept in agreement by tests that compare them byte for byte.

Everything is **little-endian** unless stated otherwise.

## 1. Conventions

* `u8`, `u16`, `u32`, `u64` are unsigned integers in little-endian order.
* `varint` is an unsigned LEB128 integer (see §2.2).
* "Must" is normative. "Should" is a recommendation a conforming decoder may
  ignore; a decoder that ignores one of these still decodes correctly.
* A conforming decoder **must reject** any stream that violates a "must". It
  must not attempt to decode a stream it has rejected, and must not produce
  output for one. Returning a *plausible but wrong* result is the single
  worst failure a decoder can have, because nothing downstream can detect it.

## 2. Primitives

### 2.1 Integers

Fixed-width integers are used wherever the value's maximum is known in advance,
because it lets a decoder bounds-check before allocating. A decoder must reject a
stream whose declared sizes exceed its own limits rather than attempting the
allocation.

### 2.2 `varint`

Unsigned LEB128, least-significant group first:

```text
byte := group(7 bits) | continuation bit (0x80)
value := group_0 | (group_1 << 7) | (group_2 << 14) | ...
```

The final byte has the continuation bit clear. A decoder must reject a `varint`
that is longer than 10 bytes, or whose 10th byte carries more than one
significant bit, since neither can represent a `u64`.

### 2.3 Hashes

Where a hash is present it is a **32-byte BLAKE3 digest**, unkeyed, of the
decoded content of the frame (or of the whole stream, for the end marker).
Hashes are compared for equality; they are not used as a MAC and provide no
authentication.

## 3. Stream grammar

```text
Stream     := ContainerHeader Frame* EndMarker
Frame      := FrameHeader Payload [ContentHash]
EndMarker  := EndMagic ContentSize [ContentHash]
```

A stream is a container header, zero or more frames, and an end marker. The
end marker is **mandatory**: a stream that ends without one is truncated, not
complete, and a conforming decoder must report it as such. This is what makes
truncation detectable — a stream cut short anywhere before the end marker is
rejected rather than silently yielding a prefix.

## 4. Container header

28 bytes, fixed.

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 4 | `magic` | `1A 43 5A 4F` |
| 4 | 1 | `version_major` | 2 for this document |
| 5 | 1 | `version_minor` | 0 for this document |
| 6 | 2 | `header_len` | 28 |
| 8 | 4 | `flags` | see below |
| 12 | 4 | `chunk_size` | target uncompressed bytes per frame, > 0 |
| 16 | 4 | `window_size` | maximum match distance, > 0, `<= chunk_size` |
| 20 | 8 | `content_size` | total uncompressed bytes, or 0 if unknown |

`magic` is the DOS end-of-file byte `0x1A` followed by `CZO`. The leading `0x1A`
means a stream damaged by a text-mode transfer or a truncated download is
unlikely to be mistaken for valid.

### 4.1 Version handling

`version_major` must be understood before anything else is read. A decoder that
does not implement the major version must reject the stream — it must not
attempt to decode it "as far as it can".

`version_minor` is informational. A decoder implementing major version 1 must
accept any minor version.

`header_len` gives the total size of the header, allowing a later minor version to
append fields that older decoders skip. A decoder must:

* reject `header_len` outside `8 ..= 4096`;
* reject the stream if fewer than `header_len` bytes are available;
* skip the bytes between the end of the fields it knows and `header_len` without
  interpreting them.

A decoder should warn, but must not fail, when it skips bytes.

### 4.2 Flags

| Bit | Name | Meaning |
|---|---|---|
| 0 | `CONTENT_SIZE` | `content_size` in the header is meaningful |
| 1 | `CONTENT_CHECKSUM` | the end marker carries a whole-content hash |
| 2 | `CHUNK_CHECKSUM` | every frame carries a content hash |

A decoder must reject a stream with any other bit set. Flags are deliberately
**not** forward compatible: an encoder may only set bits defined by the major
version it writes, which is what lets a reader treat an unknown bit as a hard
error instead of guessing.

When `CONTENT_SIZE` is clear, `content_size` must be written as 0 and a decoder
must not use the field. This is the normal case for streamed input, where the
total is not known until the end.

### 4.3 Field validation

A decoder must reject the stream if `chunk_size == 0`, if `window_size == 0`, or
if `window_size > chunk_size`. The last is not merely untidy: frames never exceed
`chunk_size`, so a larger window is unreachable and would only waste decoder
memory.

## 5. Frame header

16 bytes, fixed, immediately followed by `payload_size` payload bytes.

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 4 | `magic` | `46 52 4D 1A` (`FRM` + `0x1A`) |
| 4 | 1 | `pipeline` | see §5.1 |
| 5 | 1 | `transform` | see §5.2 |
| 6 | 1 | `flags` | see §5.3 |
| 7 | 1 | `dict_id` | 0 = none, else 1-based dictionary ordinal |
| 8 | 4 | `content_size` | bytes this frame decodes to |
| 12 | 4 | `payload_size` | bytes of payload following the header |

`content_size` is the number of bytes the frame **decodes to**, which is what lets
a decoder pre-allocate exactly and, more importantly, refuse to write more than
the frame claims. A decoder must reject a frame whose decoding would produce
fewer or more than `content_size` bytes.

A frame occupies `16 + payload_size` bytes, plus a 32-byte content hash if
`CHUNK_CHECKSUM` is set in the container header.

### 5.1 Pipeline ids

| Id | Name | Content |
|---|---|---|
| 0 | reserved | invalid in a frame; the end marker uses its own magic |
| 1 | `store` | payload bytes *are* the content |
| 2 | `lz-fast` | byte-oriented LZ77 token stream (§7.2) |
| 3 | `statistical` | LZ77 + context model + range coder (§7.3) |
| 4 | `rle` | run-length groups (§7.1) |
| 5 | `dictionary` | content that defines a dictionary for later frames |

Ids are stable and never reused; new ids are appended. A decoder must reject an
unknown id rather than guessing. `pipeline == 0` is always invalid.

For `store`, a decoder must reject a frame where `payload_size != content_size`.

For `dictionary`, a decoder must reject a frame whose `dict_id != 0`: a
dictionary frame is by definition a standalone artifact, and letting it depend on
a dictionary would be a cycle.

### 5.2 Transform ids

| Id | Name | Meaning |
|---|---|---|
| 0 | `none` | identity |
| 1 | `delta` | byte-wise difference against the previous decoded byte |

The transform is applied *before* the pipeline on encode and inverted *after* the
pipeline on decode. The first byte of a `delta` frame's content is encoded as its
own difference, i.e. against zero, so the transform is reversible without
carrying state. A decoder must reject an unknown id.

### 5.3 Frame flags

| Bit | Name | Meaning |
|---|---|---|
| 0 | `INDEPENDENT` | the frame references only bytes it produces itself |

`INDEPENDENT` means the frame can be decoded without any other frame, in any
order, on any number of threads. A decoder may exploit this to parallelise; it
must not assume it.

A decoder must reject any other bit set, for the same reason as the container
flags.

### 5.4 Dictionaries

`dict_id` refers to a preceding `dictionary` frame by 1-based ordinal, counting
such frames from the start of the stream. Dictionary priming is not yet emitted
by this implementation — every frame carries `dict_id = 0` — but the field and
its rules are part of the format so that adding it later is not a breaking change.

## 6. End marker

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 4 | `magic` | `45 4E 44 1A` (`END` + `0x1A`) |
| 4 | 8 | `content_size` | total bytes decoded across all frames |
| 12 | 32 | `content_hash` | present iff `CONTENT_CHECKSUM` is set |

The 4-byte magic is what distinguishes the end marker from a frame without
reading either one's length first.

A conforming decoder must:

* require the end marker, and reject a stream that ends without one;
* reject `content_size` disagreeing with the total it decoded;
* when `CONTENT_CHECKSUM` is set, verify the hash and reject a mismatch.

`content_size` in the end marker is authoritative even when the container
header's `CONTENT_SIZE` flag is clear, because by this point the total is always
known. This is the check that turns a silent truncation into an error.

## 7. Pipelines

A decoder dispatches on `pipeline` and needs nothing outside the frame — except
`dict_id`, which this version never sets.

### 7.1 `rle` (id 4)

A sequence of two-byte-headed groups:

```text
control := 0x80 | (n - 1)   followed by one byte   -> a run of n copies
control := 0x00 | (n - 1)   followed by n bytes    -> n literal bytes
```

`n` is therefore in `1 ..= 128`. The high bit selects the group kind.

Decoding ends when `content_size` bytes have been produced; there is no
terminator. A decoder must reject a group that would overrun `content_size`, a
run longer than the remaining space, or a literal group whose bytes are missing.

### 7.2 `lz-fast` (id 2)

A sequence of blocks:

```text
<varint literal_count>  <literal_count bytes>
<varint offset - 1>     <varint match_length - 4>     (absent in the last block)
```

The last block is marked by setting bit 63 of its `literal_count` varint, and
carries no match fields. Bit 63 is otherwise free because a literal count never
approaches `2^63`. A conforming encoder must always emit this terminating block,
even when it has zero literals.

A decoder must reject a stream whose terminating block does not land exactly on
`content_size`, an offset of zero or greater than the bytes decoded so far, an
offset greater than `window_size`, or a match that would overrun the frame.

**Overlapping matches.** When `offset < match_length`, the copy must run forward
one byte at a time, so each byte sees the value the byte before it just wrote.
This extends a repeating period — with the literals `abcdef`, an offset of 3 and a
length of 12 yields `abcdefdefdefdefdef`. A decoder that uses memmove semantics
here, or copies the source region in one block, will read bytes that have not been
written yet and produce wrong output on any input where the overlapped bytes
differ. This is the single easiest way to get a subtly broken LZ decoder.

`offset` is stored minus one so that distance zero is impossible and remains
reserved as "no match".

### 7.3 `statistical` (id 3)

Payload layout:

```text
<varint coded_len> <coded_len bytes of range-coded data> <model tables>
```

The decoder reads `coded_len` first so it can delimit the coded section without
parsing anything else.

#### 7.3.1 Symbols

The coded section is a sequence of symbols against three kinds of model. All
lengths and offsets are coded as *symbols*, never spliced in as raw bits, so the
coder keeps a single read position. (A design that mixed raw bits into the coded
stream creates a second position that can desynchronise, and a desynchronised
frame decodes to plausible-but-wrong bytes rather than failing.)

*Sequence symbols* (`0 ..= 0x7FF`):

```text
bit 10      a match follows the literal run
bits 8..=9  offset width class
bits 4..=7  match length field
bits 0..=3  literal run length field
```

A length field of 15 means the real value is carried by a following *extra*
symbol. Offsets are split by magnitude into a width class and low-order bytes,
coded as extra symbols against a uniform 256-way model. The high bits of the
class are unused; a decoder must reject a sequence symbol with any of them set.

The *literal* run length escape is a varint: seven payload bits per symbol, low
group first, with the top bit set on every symbol except the last. It must be
read to its terminator, and it must carry the length for **every** value the
field can describe — a fixed-width escape is not permitted, because the literal
run length is a `u32` and a short escape silently truncates it, which
desynchronises the range coder and makes the frame undecodable rather than merely
larger. A decoder must reject an escape whose groups exceed a `u32`, and must
reject one that never terminates.

The *match* length escape is a single symbol and does need a bound: it is
written as `mlen - 4`, so with `MAX_MATCH = 258` it is at most 254 and always
fits. The offset width class is likewise exact, because an offset never exceeds
the chunk size and so needs at most four bytes.

*Literal symbols* are one per literal byte, drawn from one of 16 order-1
contexts keyed on the previous literal's high nibble. The first literal of a run
uses context 0.

#### 7.3.2 Model tables

What is transmitted is the **training counts**, not the frequencies. Frequencies
are a pure function of the counts and both sides compute them identically, so
sending frequencies would send a redundant derivation that is also far larger: a
dense encoding of these tables costs about 12 KiB per frame, while the count
encoding costs a few hundred bytes on real data.

```text
<varint alphabet> <varint present> ( <varint delta> <varint count> ) x present
```

`delta` is one more than the gap from the previous symbol, the first being
measured from zero, so consecutive observed symbols cost one byte however deep in
the alphabet they sit. The table begins with the sequence model, followed by
exactly 16 literal-context tables in context order.

Frequencies are derived as follows. Let `n` be the alphabet size and `S` the sum
of the counts. Every symbol is given one unit; the remaining `TOTAL - n` units are
divided among the observed symbols in proportion to their counts, with the integer
remainder added to the least significant symbols. `TOTAL` is 65536.

Reserving the floor this way is not an optimisation, it is required for
correctness of the model: a naive normalisation scales the floor by the mass
ratio, which hands most of the probability mass to symbols the chunk never
contained.

A decoder must reject a table whose `alphabet` is 0 or above the coder's limit,
whose `present` exceeds `alphabet`, whose deltas are zero, or whose symbol
positions fall outside the alphabet. A zero stored count is non-canonical,
because the encoder never writes one.

#### 7.3.3 Range coder

The coded section is constriction's ANS range coder, 16-bit precision, with
`coded_len` payload bytes preceded by a `u32` word count. Symbols and models are
described in the reference implementation; a third-party decoder must reproduce
this exact coder to read the payload.

## 8. Transforms

`none` is the identity. `delta` replaces each byte with its difference from the
previous byte, wrapping modulo 256, with the first byte differenced against zero.
Inversion is a running sum modulo 256.

## 9. Integrity and damage reporting

The format is built so that damage produces an error rather than wrong bytes:

* Structural fields are range-checked before use (`chunk_size`, `window_size`,
  `header_len`, alphabet sizes, frame counts).
* Frame decoders are bounded by `content_size` and must refuse to write past it.
* Content hashes, when enabled, are verified per frame and over the whole stream.
* The mandatory end marker makes truncation detectable anywhere in the stream.

A decoder must distinguish these cases rather than collapsing them into "failed":
wrong magic, unsupported major version, truncated input, structurally corrupt
input, and checksum mismatch are different diagnoses with different remedies, and
conflating them makes a damaged file undiagnosable in practice.

## 10. Limits

| Quantity | Limit | Reason |
|---|---|---|
| Chunk size | 1 GiB | a decoder must hold a whole chunk; see `WindowConfig::MAX_CHUNK` |
| Container header | 4096 bytes | bounds `header_len` |
| Model alphabet | 4096 symbols | bounds the count table |
| `content_size` per frame | `u32` | the field is 4 bytes |
| Total content | `u64` | the end marker field is 8 bytes |
| `TOTAL` frequency mass | 65536 | 16-bit precision, so a `u16` table encoding is exact |

The chunk limit is far below the 4 GiB the `u32` field could express, on purpose:
a decoder allocates a whole chunk before writing any of it, and an engine that
promises bounded memory has to mean it.

## 11. Version history

| Version | Change |
|---|---|
| 2.0 | `statistical`: the escaped literal-run length is a varint (7 payload bits per byte, top bit = continuation) instead of a fixed two bytes. The old encoding could only express lengths up to 65 535, so any literal run at or above 65 536 was truncated on the way out and the frame could not be decoded. |
| 1.0 | Initial format: container header, frames, end marker, pipelines `store`, `lz-fast`, `statistical`, `rle`, `dictionary`; transforms `none`, `delta`. |

Any change to the meaning of an existing field, or to a pipeline's payload, is a
major version. Adding a pipeline or transform id, or a new minor-version header
field, is minor.
