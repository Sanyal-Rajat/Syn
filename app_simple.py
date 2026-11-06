"""Results-only Streamlit presentation for a nontechnical audience."""
import json
import logging
import os
from pathlib import Path
from urllib.parse import quote

import httpx
import streamlit as st

from lidarpoc.presentation import (CHARTS, chart_caption, chart_spec, gallery, image_name,
                                  overlap_text, presentation_image, share_html, summary)

LOGGER = logging.getLogger(__name__)

st.set_page_config(page_title='Scene comparison', layout='wide')
st.markdown('<style>.block-container{max-width:1500px;padding-top:2rem}</style>', unsafe_allow_html=True)
st.title('Scene comparison')
st.write('See how the generated versions compare with a real scene.')
st.caption('The generated versions show coloured scene points on recorded photos.')
present_image = st.cache_data(ttl=300, max_entries=64, show_spinner=False)(presentation_image)


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
    source = st.radio('Open results from', ['Online results', 'Saved results'],
                      index=1 if os.environ.get('LIDARPOC_RESULTS') else 0)
    if st.button('Refresh results'):
        remote_json.clear()
        remote_image.clear()
        present_image.clear()
        st.session_state.pop('simple_share', None)
        st.rerun()
    with st.expander('Connection settings'):
        endpoint = st.text_input('Results address', 'http://127.0.0.2:9000').rstrip('/')
        folder = st.text_input('Results folder', os.environ.get('LIDARPOC_RESULTS', str(Path.home() / 'lidarpoc')),
                               help='Choose the folder containing your downloaded results.')

try:
    if source == 'Online results':
        jobs = [row for row in remote_json(endpoint, '/jobs') if row.get('action') == 'generate' and row.get('status') == 'completed']
        if not jobs:
            st.info('No results are ready yet.')
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
    LOGGER.warning('Unable to open scene results: %s', error)
    st.info('These results could not be opened. Check your connection or choose another saved folder.')
    st.stop()

if not isinstance(report, dict) or report.get('action') != 'generate':
    st.info('Choose a finished scene result.')
    st.stop()

if source == 'Saved results' and not report.get('pixel_analysis', {}).get('frames'):
    analysis_path = directory / 'pixel_graphs.json'
    if analysis_path.is_file():
        try:
            report = {**report, 'pixel_analysis': json.loads(analysis_path.read_text())}
            identity += (analysis_path.stat().st_mtime_ns,)
        except (OSError, ValueError) as error:
            LOGGER.warning('Unable to read saved scene measurements: %s', error)
            st.info('The saved graph results could not be opened. Available images are shown below.')

identity = ('comparison-bars',) + identity
data = summary(report)
st.subheader('How close are the versions to the real scene?')
st.write(overlap_text(data))
if not data['paired']:
    st.info('These results show individual versions. Select a comparison result to see Without training and With training together.')
for key, info in CHARTS.items():
    if data[key]:
        st.vega_lite_chart(chart_spec(data[key], info['title'], info['unit'], info['higher_better']),
                           use_container_width=True)
        st.caption(chart_caption(data, key))
    else:
        st.info(info['title'] + ' is unavailable for this result.')
with st.expander('How to read the graphs'):
    st.write('Coverage match: higher means closer coverage of the real scene. '
             '100% means the same coverage.')
    st.write('Distance difference: lower means a closer pattern of distances. '
             'Zero means the same distance pattern.')

frames = gallery(report)
if not frames:
    st.info('No camera photos are available for these results.')
    reason = report.get('camera_overlays', {}).get('reason')
    if reason:
        LOGGER.warning('Camera images unavailable: %s', reason)

for index, frame in enumerate(frames, 1):
    st.subheader(f'View {index}')
    for start in range(0, len(frame['cards']), 4):
        cards = frame['cards'][start:start + 4]
        for panel, card in zip(st.columns(len(cards)), cards):
            panel.markdown('**' + card['label'] + '**')
            try:
                if not card.get('file'):
                    raise ValueError('No matching image was recorded')
                original = load_image(card['file'])
                card['data'] = present_image(original, card['label']) if card['overlay'] else original
                panel.image(card['data'])
            except (httpx.HTTPError, OSError, ValueError) as error:
                LOGGER.warning('Unable to display scene image: %s', error)
                panel.info('Image unavailable')
                card['data'] = None

if frames or data['overlap'] or data['wasserstein']:
    if st.button('Create shareable summary'):
        st.session_state['simple_share'] = (identity, share_html(report, data, frames))
    prepared = st.session_state.get('simple_share')
    if prepared and prepared[0] == identity:
        st.download_button('Download summary', prepared[1], 'scene_comparison.html', 'text/html')
