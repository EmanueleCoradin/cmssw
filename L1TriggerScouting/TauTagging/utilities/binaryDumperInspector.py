"""
Format of each event:
  [64-bit header][payload word 0][payload word 1]...[payload word N-1]

Header bit header:
  bits 63-62 : must be binary 10 for a valid event header
  bit 61     : error bit
  bits 60-56 : local run number
  bits 55-24 : orbit number
  bits 23-12 : bunch crossing number - 1  (stored range: 0..3563)
  bits 11-0  : number of 64-bit payload words following the header

The script:
  1. Parses all events from the raw file.
  2. Verifies that events are arranged in 3564-event blocks with identical orbit.
  3. Optionally checks that BX inside each block is ordered from 1 to 3564.
  4. Writes one output file per orbit containing the original raw bytes
     of all events belonging to that orbit.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import BinaryIO, Iterator, Optional, Sequence
import math
import argparse

import awkward as ak

BX_PER_ORBIT = 3564
VALID_HEADER_TAG = 0b10

@dataclass
class Event:
    index: int
    file_offset: int
    header_word: int
    raw_bytes: bytes
    error_bit: int
    local_run: int
    orbit: int
    bx_stored: int
    bx: int
    num_payload_words: int

@dataclass
class Header:
    error_bit: int
    local_run: int
    orbit: int
    bx_stored: int
    bx: int
    num_payload_words: int

@dataclass
class Candidate:
    pt: float
    eta: float
    phi: float

class ParseError(Exception):
    pass

def parse_header(header_word: int) -> dict:
    """Decode a 64-bit event header."""
    tag = (header_word >> 62) & 0b11
    if tag != VALID_HEADER_TAG:
        raise ParseError(
            f"Invalid header tag: bits[63:62] = {tag:02b}, expected 10"
        )

    error_bit = (header_word >> 61) & 0x1
    local_run = (header_word >> 56) & 0x1F
    orbit = (header_word >> 24) & 0xFFFFFFFF
    bx_stored = (header_word >> 12) & 0xFFF
    num_payload_words = header_word & 0xFFF

    if bx_stored > 3563:
        raise ParseError(
            f"Invalid BX field: stored value {bx_stored} is outside 0..3563"
        )

    bx = bx_stored + 1

    return Header(
        error_bit=error_bit,
        local_run=local_run,
        orbit=orbit,
        bx_stored=bx_stored,
        bx=bx,
        num_payload_words=num_payload_words
    )

def parse_candidate(candidate_word: int) -> dict:
    """Decode a 64-bit candidate word."""
    # pT: unsigned 14-bit integer
    pt_int = candidate_word & 0x3FFF
    pt = pt_int * 0.25

    # eta: signed 12-bit integer stored in bits [25:14]
    eta_int = (candidate_word >> 14) & 0xFFF
    if eta_int & 0x800:
        eta_int -= 0x1000
    eta = eta_int * (math.pi / 720.0)

    # phi: signed 11-bit integer stored in bits [36:26]
    phi_int = (candidate_word >> 26) & 0x7FF
    if phi_int & 0x400:
        phi_int -= 0x800
    phi = phi_int * (math.pi / 720.0)

    return Candidate(
        pt=pt,
        eta=eta,
        phi=phi
    )

def iter_events(stream: BinaryIO, endian: str) -> Iterator[Event]:
    """Yield events from the binary stream."""
    if endian == "little":
        unpack_u64 = struct.Struct("<Q").unpack
    elif endian == "big":
        unpack_u64 = struct.Struct(">Q").unpack
    else:
        raise ValueError(f"Unsupported endian: {endian}")

    event_index = 0

    while True:
        file_offset = stream.tell()
        header_bytes = stream.read(8)

        if not header_bytes:
            break

        if len(header_bytes) != 8:
            raise ParseError(
                f"Truncated file: incomplete 64-bit header at offset {file_offset}"
            )

        header_word = unpack_u64(header_bytes)[0]
        header = parse_header(header_word)

        payload_nbytes = header.num_payload_words * 8
        payload_bytes = stream.read(payload_nbytes)

        if len(payload_bytes) != payload_nbytes:
            raise ParseError(
                "Truncated file: "
                f"event {event_index} at offset {file_offset} declares "
                f"{header.num_payload_words} payload words "
                f"({payload_nbytes} bytes), but file ends early"
            )

        yield Event(
            index=event_index,
            file_offset=file_offset,
            header_word=header_word,
            raw_bytes=header_bytes + payload_bytes,
            error_bit=header.error_bit,
            local_run=header.local_run,
            orbit=header.orbit,
            bx_stored=header.bx_stored,
            bx=header.bx,
            num_payload_words=header.num_payload_words,
        )

        event_index += 1


def verify_blocks(
    events: list[Event],
    check_bx: bool = True,
    allow_incomplete_last_block: bool = False,
) -> None:
    """Verify that events come in 3564-event blocks with identical orbit numbers."""
    n_events = len(events)

    if n_events == 0:
        raise ParseError("No events found in input file")

    for block_start in range(0, n_events, BX_PER_ORBIT):
        block_end = min(block_start + BX_PER_ORBIT, n_events)
        block = events[block_start:block_end]

        if len(block) != BX_PER_ORBIT and not allow_incomplete_last_block:
            raise ParseError(
                f"Last block is incomplete: block starting at event {block_start} "
                f"contains {len(block)} events instead of {BX_PER_ORBIT}"
            )

        first_orbit = block[0].orbit
        for ev in block:
            if ev.orbit != first_orbit:
                raise ParseError(
                    "Orbit consistency error inside 3564-event block: "
                    f"block starting at event {block_start}, "
                    f"expected orbit {first_orbit}, found orbit {ev.orbit} "
                    f"at event {ev.index} (file offset {ev.file_offset})"
                )

        if check_bx:
            for i, ev in enumerate(block):
                expected_bx = i + 1
                if ev.bx != expected_bx:
                    raise ParseError(
                        "BX ordering error inside orbit block: "
                        f"block starting at event {block_start}, orbit {first_orbit}, "
                        f"event {ev.index} has BX={ev.bx}, expected {expected_bx}"
                    )

import struct
from dataclasses import fields, replace
from typing import Sequence

def merge_events(
    events: Sequence[Event],
    endian: str = "little",
) -> Event:
    if not events:
        raise ValueError("At least one event is required")

    reference = events[0]

    # Check that all inputs correspond to the same event.
    for event in events[1:]:
        if (event.orbit, event.bx) != (reference.orbit, reference.bx):
            raise ValueError(
                "Events do not match: "
                f"{(reference.orbit, reference.bx)} != "
                f"{(event.orbit, event.bx)}"
            )

    total_payload_words = sum(
        event.num_payload_words for event in events
    )

    # The header reserves 12 bits for the payload-word count.
    if total_payload_words > 0xFFF:
        raise ValueError(
            f"Merged payload contains {total_payload_words} words, "
            "but the header can represent at most 4095"
        )

    if endian == "little":
        unpack_u64 = struct.Struct("<Q").unpack
        pack_u64 = struct.Struct("<Q").pack
    elif endian == "big":
        unpack_u64 = struct.Struct(">Q").unpack
        pack_u64 = struct.Struct(">Q").pack
    else:
        raise ValueError(f"Unsupported endian: {endian}")

    payloads: list[bytes] = []

    for event in events:
        expected_size = 8 * (event.num_payload_words + 1)

        if len(event.raw_bytes) != expected_size:
            raise ValueError(
                f"Event {event.index} has {len(event.raw_bytes)} bytes, "
                f"expected {expected_size}"
            )

        # Remove the 8-byte header.
        payloads.append(event.raw_bytes[8:])

    # Replace only the lowest 12 bits of the reference header.
    merged_header_word = (
        (reference.header_word & ~0xFFF)
        | total_payload_words
    )

    merged_raw_bytes = (
        pack_u64(merged_header_word)
        + b"".join(payloads)
    )

    return replace(
        reference,
        header_word=merged_header_word,
        raw_bytes=merged_raw_bytes,
        num_payload_words=total_payload_words,
    )

def merge_event_lists(
    *event_lists: Sequence[Event],
    endian: str = "little",
) -> list[Event]:
    if not event_lists:
        return []

    return [
        merge_events(events, endian=endian)
        for events in zip(*event_lists, strict=True)
    ]

def get_arrays(
    events: list[Event],
    endian: str,
) -> ak.Array:
    if not events:
        raise ParseError("No events found in input file")

    byte_order = "<" if endian == "little" else ">"

    records = []

    for ev in events:
        expected_nbytes = 8 * (ev.num_payload_words + 1)

        if len(ev.raw_bytes) != expected_nbytes:
            raise ParseError(
                f"Event {ev.index}: raw byte size mismatch: "
                f"got {len(ev.raw_bytes)}, "
                f"expected {expected_nbytes}"
            )

        unpack_u64 = struct.Struct(
            f"{byte_order}{ev.num_payload_words + 1}Q"
        ).unpack

        all_words = unpack_u64(ev.raw_bytes)
        payload_words = all_words[1:]

        ev_dict = {
            "orbitNumber": ev.orbit,
            "bunchCrossing": ev.bx,
            "L1PF_pt": [],
            "L1PF_eta": [],
            "L1PF_phi": [],
        }

        for word in payload_words:
            candidate = parse_candidate(word)
            ev_dict["L1PF_pt"].append(candidate.pt)
            ev_dict["L1PF_eta"].append(candidate.eta)
            ev_dict["L1PF_phi"].append(candidate.phi)

        records.append(ev_dict)

    return ak.Array(records)

def summarize(events: list[Event]) -> str:
    """Return a compact textual summary."""
    n_events = len(events)
    n_orbits = len({ev.orbit for ev in events})
    n_error = sum(ev.error_bit for ev in events)
    first = events[0]
    last = events[-1]

    return (
        f"Parsed events      : {n_events}\n"
        f"Distinct orbits    : {n_orbits}\n"
        f"Events with error  : {n_error}\n"
        f"First orbit / BX   : {first.orbit} / {first.bx}\n"
        f"Last orbit / BX    : {last.orbit} / {last.bx}\n"
    )

def inspect_dump(
    input_file: Path, 
    endian: str = "little", 
    check_bx: bool = True
) -> ak.Array:
    if not input_file.is_file():
        print(f"Error: input file not found: {input_file}", file=sys.stderr)
        return 1

    try:
        with open(input_file, "rb") as fh:
            events = list(iter_events(fh, endian=endian))

        verify_blocks(
            events,
            check_bx=check_bx,
        )

        arrays = get_arrays(events, endian=endian)
        return arrays

    except (ParseError, FileExistsError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    # define file paths
    paths = [
        Path("/eos/user/g/gizago/store/ScoutPFTaus/emulation/signal/PF_barrel_SoftTaus_17_0_X.dump"), 
        Path("/eos/user/g/gizago/store/ScoutPFTaus/emulation/signal/PF_hgcal_SoftTaus_17_0_X.dump")
    ]

    # get events for each path
    event_lists = []

    for p in paths:
        with open(p, "rb") as f:
            events = list(iter_events(f, endian="little"))
            verify_blocks(
                events,
                check_bx=True,
            )
            event_lists.append(events)

    # merge events 
    merged_events = merge_event_lists(*event_lists)

    # get arrays from merged events
    arrays = get_arrays(merged_events, endian="little") 

    # create root file
    import uproot
    file = uproot.recreate("softTauNano-17_0_X-binaries.root")
    file.mktree("Events", {key: arrays[key] for key in arrays.fields})
    file.close()