"""Results-only Streamlit presentation for a nontechnical audience."""
import json
import os
from pathlib import Path
from urllib.parse import quote

import httpx
import streamlit as st

from lidarpoc.presentation import chart, gallery, image_name, overlap_text, share_html, summary

st.set_page_config(page_title='See what training changes', layout='wide')
st.markdown('<style>.block-container{max-width:1500px;padding-top:2rem}</style>', unsafe_allow_html=True)
st.title('See what training changes')
st.write('Compare the untrained and trained versions on the same scene.')
st.caption('Photos are recorded. The coloured laser points are synthetic.')


@st.cache_data(ttl=300, show_spinner=False)
def remote_json(endpoint, path):
    response = httpx.get(endpoint + path, timeout=30, trust_env=False)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=300, show_spinner=False)
def remote_image(endpoint, job_id, filename):
    response = httpx.get(endpoint + f'/jobs/{job_id}/overlays/' + quote(image_name(filename), safe=''),
                         timeout=30, trust_env=False)
    response.raise_for_status()
    return response.content


with st.sidebar:
    st.subheader('Choose results')
    source = st.radio('Open results from', ['Connected server', 'Downloaded folder'],
                      index=1 if os.environ.get('LIDARPOC_RESULTS') else 0)
    if st.button('Refresh results'):
        remote_json.clear()
        remote_image.clear()
        st.session_state.pop('simple_share', None)
        st.rerun()
    with st.expander('Setup'):
        endpoint = st.text_input('Server connection', 'http://127.0.0.2:9000').rstrip('/')
        folder = st.text_input('Downloaded results folder', os.environ.get('LIDARPOC_RESULTS', str(Path.home() / 'lidarpoc')),
                               help='Folder containing result.json and the overlays folder.')
        st.caption('Uses your existing SSH tunnel. The technical app remains available as app.py.')

try:
    if source == 'Connected server':
        jobs = [row for row in remote_json(endpoint, '/jobs') if row.get('action') == 'generate' and row.get('status') == 'completed']
        if not jobs:
            st.info('No finished scene results are available yet.')
            st.stop()
        identifiers = [row['id'] for row in jobs]
        requested = st.query_params.get('job')
        job_id = st.sidebar.selectbox('Finished result', identifiers,
            index=identifiers.index(requested) if requested in identifiers else 0,
            format_func=lambda value: 'Most recent result' if value == identifiers[0] else f'Result {identifiers.index(value) + 1}')
        report = remote_json(endpoint, f'/jobs/{job_id}/result')
        identity = (endpoint, job_id)

        def load_image(filename):
            return remote_image(endpoint, job_id, filename)
    else:
        directory = Path(folder).expanduser().resolve()
        result_path = directory / 'result.json'
        report = json.loads(result_path.read_text())
        identity = (str(directory), result_path.stat().st_mtime_ns)

        def load_image(filename):
            path = (directory / 'overlays' / image_name(filename)).resolve()
            if path.parent != (directory / 'overlays').resolve():
                raise ValueError('Image must be inside the downloaded overlays folder')
            return path.read_bytes()
except (httpx.HTTPError, OSError, ValueError, KeyError) as error:
    st.info('These results could not be opened. Check the selected folder or reconnect the existing server tunnel.')
    with st.expander('Connection details'):
        st.text(str(error))
    st.stop()

if not isinstance(report, dict) or report.get('action') != 'generate':
    st.info('Choose a completed scene-generation result.')
    st.stop()

data = summary(report)
if data['paired']:
    st.write(overlap_text(data))
    if data['overlap']:
        st.image(chart(data['overlap'], 'Scan overlap with the real recording', '%', True))
        st.caption(f"Based on {data['overlap_frames']} of {data['frame_count']} frames. Higher means more shared laser-return pixels; it is not a photo-quality score.")
    else:
        st.info('Scan-overlap measurements are missing. Refresh this run’s analysis in the technical app.')
    if data['wasserstein']:
        st.image(chart(data['wasserstein'], 'Distance-distribution difference (Wasserstein-1)', 'm'))
        st.caption(f"Based on {data['wasserstein_frames']} of {data['frame_count']} frames. A smaller bar means the spread of measured distances is closer to the real recording. Each frame is counted equally.")
    else:
        st.info('The Wasserstein measurement is missing. Use Build / refresh pixel and OT graphs in the technical app.')
    with st.expander('What do the two charts mean?'):
        st.write('Scan overlap compares which laser-return pixels are present. A value of 100% means the same occupied scan pixels. '
                 'Wasserstein-1 compares the distributions of measured distances in metres, using all valid values. '
                 'The first chart pools scan pixels across shared valid frames; the second averages per-frame distances. '
                 'The real reference is 100% overlap and zero distance from itself. '
                 'Both synthetic versions use the same scene geometry, so these charts describe two aspects of the scan rather than overall photographic realism.')
else:
    st.info('This result does not contain a verified untrained/trained pair. Its available images are shown below.')
    with st.expander('Choose a result with both versions'):
        st.write('In app.py, generate a run with Compare without training / with training enabled, then select that completed result here.')

frames = gallery(report)
if not frames:
    st.info('This result has no matched camera images available.')
    reason = report.get('camera_overlays', {}).get('reason')
    if reason:
        with st.expander('Image availability'):
            st.text(reason)

for index, frame in enumerate(frames, 1):
    st.subheader(f'Scene view {index}')
    for start in range(0, len(frame['cards']), 4):
        cards = frame['cards'][start:start + 4]
        for panel, card in zip(st.columns(len(cards)), cards):
            panel.markdown('**' + card['label'] + '**')
            try:
                if not card.get('file'):
                    raise ValueError('No matching image was recorded')
                card['data'] = load_image(card['file'])
                panel.image(card['data'])
            except (httpx.HTTPError, OSError, ValueError) as error:
                panel.info('Image unavailable')
                card['data'] = None
                with panel.expander('Details'):
                    st.text(str(error))

if frames:
    if st.button('Prepare a shareable summary'):
        st.session_state['simple_share'] = (identity, share_html(report, data, frames))
    prepared = st.session_state.get('simple_share')
    if prepared and prepared[0] == identity:
        st.download_button('Download image comparison', prepared[1],
                           'training_comparison.html' if data['paired'] else 'scene_images.html', 'text/html')
