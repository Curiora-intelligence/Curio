import base64
import io
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from PIL import Image
import pytest

from app.services.coaching import AudioMetrics, InterviewObservation, communication_metrics, safe_interview_text
from app.services.ephemeral import EphemeralStore
from app.services.live import LiveEvent, LiveSession, parse_observation


def jpeg():
    buffer = io.BytesIO()
    Image.new('RGB', (32, 32), 'black').save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.mark.parametrize('fail', [True, False])
async def test_frame_throttling_cleanup_and_no_llm(fail):
    paths = []
    def vision(**kwargs):
        path = Path(kwargs['image_path'])
        paths.append(path)
        assert path.is_file()
        if fail:
            raise RuntimeError('test')
        return '{"objects":["tap"],"colors":[],"visible_text":[],"visible_details":["water droplets"]}'
    service = Mock(cache=EphemeralStore(), gateway=Mock(generate_vision=vision))
    session = LiveSession(service, 'alice', clock=lambda: 10)
    event = LiveEvent(type='frame', jpeg=jpeg())
    if fail:
        with pytest.raises(RuntimeError):
            await session.handle(event)
    else:
        assert (await session.handle(event))['type'] == 'observation'
    assert not paths[0].exists()
    assert (await session.handle(event))['reason'] == 'throttled'
    assert len(paths) == 1
    service.respond.assert_not_called()


async def test_live_routes_observation_and_user_to_shared_service():
    service = Mock(cache=EphemeralStore(), respond=AsyncMock(return_value=('answer', 'cid')))
    session = LiveSession(service, 'alice', clock=lambda: 10)
    await session.handle(LiveEvent(type='configure', mode='shopping'))
    session.observation = {'objects': ['bag'], 'colors': ['black']}
    session.observed_at = 9
    result = await session.handle(LiveEvent(type='transcript', text='Would I like this?'))
    args = service.respond.call_args.kwargs
    assert args['user_id'] == 'alice' and args['mode'] == 'shopping' and 'black' in args['context']
    assert result['conversation_id'] == 'cid'


@pytest.mark.parametrize('label', ['anxious', 'depressed', 'stressed', 'confident', 'deceptive', 'nervous'])
def test_interview_has_no_mental_state_output(label):
    assert label not in safe_interview_text('You looked ' + label + '.').lower()
    assert 'unavailable' in parse_observation('{"gaze_direction":"' + label + '"}', 'interview')


def test_metrics_are_observable():
    result = communication_metrics('one two three four', AudioMetrics(duration_seconds=2, speaking_seconds=1, pause_count=1, rms=[.1, .2]))
    assert result['words_per_minute'] == 120 and result['speaking_ratio'] == .5
    assert safe_interview_text('Your answer needs a clearer example.') == 'Your answer needs a clearer example.'
    with pytest.raises(ValueError):
        AudioMetrics(duration_seconds=1, speaking_seconds=2)


async def test_reject_bad_frames_and_stale_frames():
    session = LiveSession(Mock(cache=EphemeralStore()), 'alice')
    with pytest.raises(ValueError):
        await session.handle(LiveEvent(type='frame', jpeg='invalid'))
    assert (await session.handle(LiveEvent(type='frame', jpeg=jpeg(), captured_at=1)))['reason'] == 'stale'


@pytest.mark.parametrize('separate_sample', [True, False])
async def test_audio_sample_is_consumed_by_one_answer(separate_sample):
    service = Mock(cache=EphemeralStore(), respond=AsyncMock(return_value=('answer', 'cid')))
    session = LiveSession(service, 'alice')
    metrics = AudioMetrics(duration_seconds=2, speaking_seconds=1, pause_count=1, rms=[.1])
    if separate_sample:
        await session.handle(LiveEvent(type='metrics', metrics=metrics))
    await session.handle(LiveEvent(type='transcript', text='one two three four', metrics=None if separate_sample else metrics))
    assert json.loads(service.respond.call_args.kwargs['context'])['audio_metrics']['words_per_minute'] == 120
    await session.handle(LiveEvent(type='transcript', text='Ask the next question.'))
    assert json.loads(service.respond.call_args.kwargs['context'])['audio_metrics'] == {}
