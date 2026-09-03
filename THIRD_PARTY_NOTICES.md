# Third-party notices

## facebookresearch/vjepa2

`tt/functional_encoder.py` and `tt/functional_predictor.py` port model architecture and
RoPE-application logic from https://github.com/facebookresearch/vjepa2. Its original
license notice:

```
MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Model weights (`vjepa2-ac-vitg.pt`) are likewise released by Meta under this same
license. The pre-stripped checkpoint at
[`episod/vjepa2-ac-vitg-fpc64-256-droid-tt`](https://huggingface.co/episod/vjepa2-ac-vitg-fpc64-256-droid-tt)
is a bf16, inference-only-state derivative of that checkpoint — see
`scripts/strip_checkpoint.py`.
