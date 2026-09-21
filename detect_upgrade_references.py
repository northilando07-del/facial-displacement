"""Content-bound, native-resolution FAN landmarks for the supplied light set."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from PIL import Image, ImageOps, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--directory', type=Path)
    source.add_argument('--paths-json', type=Path, help='Ordered JSON array of selected image paths.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    paths = ([Path(p).resolve() for p in json.loads(args.paths_json.read_text(encoding='utf-8-sig'))]
             if args.paths_json else sorted(p.resolve() for p in args.directory.iterdir()
                   if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')))
    if not paths:
        raise ValueError('No reference images found.')
    digests = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    if args.output.exists():
        cache = json.loads(args.output.read_text(encoding='utf-8'))
        if [r['sha256'] for r in cache] == digests and [r['path'] for r in cache]==[str(p) for p in paths]:
            print('Verified existing full-resolution landmark cache.', flush=True)
            return
    import face_alignment
    import torch
    if args.device=='cuda' and not torch.cuda.is_available():args.device='cpu'
    detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
        device=args.device, flip_input=False, face_detector='sfd')
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, (path, digest) in enumerate(zip(paths, digests)):
        image = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
        attempts = []
        faces = None
        for side in (max(image.size), 1024, 768):
            detection_image = image.copy()
            detection_image.thumbnail((side, side), Image.Resampling.LANCZOS)
            faces = detector.get_landmarks(np.asarray(detection_image))
            attempts.append(dict(size=list(detection_image.size), count=0 if faces is None else len(faces)))
            print('Detection attempt', attempts[-1], flush=True)
            if faces is not None and len(faces) == 1:
                break
        if faces is None or len(faces) != 1:
            raise ValueError(f'Expected exactly one face in {path}; attempts={attempts}.')
        landmarks = (np.asarray(faces[0], dtype=float) + .5) * (np.asarray(image.size) / detection_image.size) - .5
        if landmarks.shape != (68, 2) or not np.isfinite(landmarks).all():
            raise ValueError(f'Invalid landmark detection in {path}.')
        records.append(dict(path=str(path.resolve()), sha256=digest,
                            size=list(image.size), landmarks=landmarks.tolist(),
                            detection='FAN68/SFD; scale fallback affects landmark detection only, shading uses original pixels',
                            detection_attempts=attempts))
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for i, (x, y) in enumerate(landmarks):
            draw.ellipse((x-3, y-3, x+3, y+3), fill=(255, 70, 20))
            draw.text((x+3, y), str(i), fill=(30, 160, 255))
        overlay.save(args.output.parent / f'landmarks_{index:02d}.png')
        print(f'Detected {index+1}/{len(paths)}: {path.name}, original {image.size}', flush=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
