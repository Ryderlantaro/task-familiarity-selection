# Reserve and remove only merge directories owned by this invocation.

import argparse
import json
import os
from pathlib import Path
import secrets
import shutil

MARKER = '.txm-merge-owner.json'


def reserve(path, token, protected):
    raw = Path(path).absolute()
    if raw.is_symlink():
        raise ValueError('Merge path cannot be a symlink')
    target = raw.resolve()
    forbidden = [Path('/'), Path.home(), Path.cwd(), *(Path(p).resolve() for p in protected)]
    if any(target == p or target in p.parents for p in forbidden):
        raise ValueError('Merge path contains a protected location')
    target.mkdir(parents=True, mode=0o700, exist_ok=False)
    stat = target.stat()
    payload = {'token': token, 'path': str(target), 'uid': os.getuid(),
               'device': stat.st_dev, 'inode': stat.st_ino}
    try:
        with (target / MARKER).open('x') as stream:
            os.chmod(target / MARKER, 0o600)
            json.dump(payload, stream)
    except BaseException:
        # Only an empty directory created by this call may be removed here.
        if not any(target.iterdir()):
            target.rmdir()
        raise
    return str(target)


def remove(path, token):
    target = Path(path).absolute()
    if target.is_symlink() or not target.is_dir():
        raise ValueError('Merge directory was removed or replaced')
    marker = target / MARKER
    if marker.is_symlink():
        raise ValueError('Ownership marker cannot be a symlink')
    payload = json.loads(marker.read_text())
    stat = target.stat()
    if (payload.get('path') != str(target.resolve()) or payload.get('uid') != os.getuid()
            or payload.get('device') != stat.st_dev or payload.get('inode') != stat.st_ino
            or not secrets.compare_digest(str(payload.get('token', '')), token)):
        raise ValueError('Merge directory ownership mismatch; refusing removal')
    if not shutil.rmtree.avoids_symlink_attacks:
        raise RuntimeError('Symlink-safe directory removal is unavailable')
    shutil.rmtree(target)


def main():
    parser = argparse.ArgumentParser(description='Reserve and remove only merge directories owned by this invocation.')
    parser.add_argument('action', choices=('reserve', 'remove', 'token'))
    parser.add_argument('--path')
    parser.add_argument('--token')
    parser.add_argument('--protect', nargs='*', default=[])
    args = parser.parse_args()
    if args.action == 'token':
        print(secrets.token_hex(32))
        return
    if not args.path or not args.token:
        parser.error('--path and --token are required')
    if args.action == 'reserve':
        print(reserve(args.path, args.token, args.protect))
    else:
        remove(args.path, args.token)


if __name__ == '__main__':
    main()
