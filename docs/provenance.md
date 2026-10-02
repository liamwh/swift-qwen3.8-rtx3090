# Provenance: tokenizer, chat template and draft vocabularies

Checked 2026-10-02. Full hashes are in
[tokenizer-provenance.sha256](tokenizer-provenance.sha256). The draft
vocabulary id lists are in [data/draft_vocab_ids](../data/draft_vocab_ids)
with their own `SHA256SUMS`.

## Is Swift 1.5's tokenizer the same as Swift 1.0's?

Yes, at the source. In UkisAI's own repositories
(`ukisai/Swift-Qwen3.8-27b` at `6bc57e4e`, `ukisai/Swift-1.5-Qwen3.8-27b` at
`bc7a1e10`), these files are byte-identical between 1.0 and 1.5:

- `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`
- `vocab.json`, `merges.txt`, `config.json`

`generation_config.json` differs. 1.5 adds `"min_p": 0` and
`"repetition_penalty": 1.0` next to the same sampling defaults, which are
both neutral values.

## Published and local copies

- The published Swift 1.5 fast repo (`liamwh/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast`
  at `e7067d6d`) carries UkisAI's 1.5 files unchanged, including the chat
  template.
- The published Swift 1.0 fast repo (`liamwh/Swift-Qwen3.8-27B-W4A16-syv-fast`
  at `fa592e0b`) was built from TheUnderscore's AWQ export. Its
  `tokenizer.json` and `tokenizer_config.json` are a different serialisation
  (the `model`, `pre_tokenizer` and `decoder` sections differ, the file is
  20.0 MB against 12.8 MB). They tokenise identically to UkisAI's: a test
  over the 8,000-token quote document, 20,000 random vocabulary tokens and a
  mixed code, unicode and special-token string produced identical ids for all
  four `tokenizer.json` files. Its chat template is UkisAI's 1.0 template.
- The directories the server loads differ from the published repos in one
  file. Their `chat_template.jinja` has been through
  `prepare/harden_chat_template.py` in the syv stack, which makes the
  `tool_call.arguments` loop fall back to a single `arguments` parameter when
  the value is not a mapping. The local 1.0 and 1.5 templates are identical
  to each other (`400a5632...`), and both differ from the upstream template
  (`c3cf9e34...`) only by that guard.

## Draft vocabularies

| list | ids | sha256 |
|---|---|---|
| base Qwen (`prepare/draft_vocab_ids.json` in syv) | 40,960 | `b64b6dfc...` |
| Swift 1.0 (`data/draft_vocab_ids/swift_draft_vocab_ids.json`) | 25,879 | `e30af795...` |
| Swift 1.5 (`data/draft_vocab_ids/swift15_draft_vocab_ids.json`) | 25,285 | `af978365...` |
| union of the two (`data/draft_vocab_ids/swift15_union_draft_vocab_ids.json`) | 29,670 | `06d479e1...` |

The Swift 1.0 and 1.5 lists share 21,494 ids and have a union of 29,670
(Jaccard 0.724). Each covers more than 99.7% of the same held-out Swift 1.5
tokens, so the ids private to one list carry about 0.2% of held-out tokens
at most. That is a statement about which tokens occur. It does not say the
two models assign similar probabilities to them.

The union list is the sorted union of the two Swift lists. Every draft-head
row is the matching row of the serving model's own int4 `lm_head`. It is
what `liamwh/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast` serves on `main`
(since commit `3d93b240`; later commits touch only the card). The earlier
revision, which served the Swift 1.5 list, is the tag `vocab-1.5-only`
(commit `e7067d6d`). Full hashes are in
`data/draft_vocab_ids/SHA256SUMS`.

## Quote workload audit

`bench/quote_vocab_audit.py` tokenises the exact quote document
(`build_doc(8000)`, 8,151 tokens after `strip()`) and counts tokens outside
each list. Output is in `bench/results/quote-vocab-audit.json`.

| list | tokens outside the list | distinct ids | main offenders |
|---|---|---|---|
| base Qwen (40,960) | 232 (2.85%) | 8 | ` Iterable` x111, ` telemetry` x56, ` batching` x56 |
| Swift 1.0 (25,879) | 10 (0.12%) | 3 | `fact` x6, ` DESIGN` x3 |
| Swift 1.5 (25,285) | 236 (2.90%) | 7 | `logger`, `.getLogger`, `(__`, `__)` x56 each |

Swift 1.0 int8, Qwen fast and Swift 1.5 int8 all sit behind the base Qwen
list and log identical quote counters (18,162 accepted of 18,873 drafted in
6,291 steps) although they are three different target models. That pins the
quote acceptance to the list.

## Checkpoint sizes

Total `*.safetensors` bytes in the directories the server loads, measured
with `du -bLc` on 2026-10-02.

| directory | bytes |
|---|---|
| Swift 1.0 fast | 15,847,047,104 |
| Swift 1.5 fast | 15,845,479,936 |
| Swift 1.0 int8 | 16,839,754,368 |
| Swift 1.5 int8 | 16,839,755,456 |

The fast pair differs by 1,567,168 bytes (1.5 is smaller) and the int8
pair by 1,088 bytes. Neither build is meaningfully heavier on disk.
