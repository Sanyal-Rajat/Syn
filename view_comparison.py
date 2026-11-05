"""Recompute pixel/voxel gaps and export the viewer on the compute machine, without sampling."""
import argparse
import json
from pathlib import Path

from lidarpoc.comparison import analyze
from lidarpoc.io import write_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path, help='Paired Generate job containing real/untrained/trained NPZ files')
    parser.add_argument('--voxel-size', type=float, default=.25)
    args = parser.parse_args()
    path = args.job / 'result.json'
    report = json.loads(path.read_text())
    if not report.get('comparison'):
        parser.error('Select paired untrained/trained generation first; two arbitrary trained samples are not a training comparison')
    from lidarpoc.pixel_graphs import analyze_pixels
    report['pixel_analysis'] = analyze_pixels(args.job, report)
    report['comparison'] = analyze(args.job, report, args.voxel_size)
    write_json(path, report)
    from lidarpoc.report import write_pdf
    write_pdf(path)
    print('Open:', (args.job / 'comparison.html').resolve())
