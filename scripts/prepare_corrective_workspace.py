"""Materialize the original two-root layout from a verified corrective dataset.

Existing differing files are never overwritten. This script has no network access.
"""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import zipfile

REPO=Path(__file__).resolve().parents[1]
PREFIX='post_rejection_20260926/transfer_revision/'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    expected=json.loads((REPO/'revisions/20260926/PAYLOAD_FILES.json').read_text(encoding='utf-8'))
    target=args.output.resolve()
    for entry in expected:
        name=entry['path']; path=PurePosixPath(name)
        assert not path.is_absolute() and '..' not in path.parts and '\\' not in name and ':' not in name
    with zipfile.ZipFile(args.dataset) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist())==len(set(archive.namelist()))
        prepared=[]
        for entry in expected:
            name=entry['path']; data=archive.read(name)
            assert len(data)==entry['bytes'] and sha(data)==entry['sha256'],name
            dest=(target/name).resolve()
            assert dest.is_relative_to(target)
            if dest.exists():
                assert dest.is_file() and sha(dest.read_bytes())==entry['sha256'],'Different existing file: '+name
            else:
                prepared.append((dest,data))
        for dest,data in prepared:
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(data)
    # Frozen scripts are archived in the repository for review, but run from
    # this materialized directory so their historical relative paths still work.
    for path in (REPO/'revisions/20260926').rglob('*'):
        if not path.is_file() or path.name=='PAYLOAD_FILES.json':
            continue
        relative=path.relative_to(REPO/'revisions/20260926').as_posix()
        dest=target/PREFIX/relative
        assert dest.exists() and sha(dest.read_bytes())==sha(path.read_bytes()),relative
    result={'status':'verified','files':len(expected),'workspace':str(target),
            'revision_directory':str(target/PREFIX),'network_used':False}
    (target/'WORKSPACE_CHECK.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
