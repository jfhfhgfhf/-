import os
import yt_dlp
import librosa
import music21
import numpy as np
from music21 import tempo, stream, metadata, meter, note, chord, clef
from PIL import Image
import io
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 업로드 폴더가 없으면 생성
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def download_youtube_audio(url, output_path='temp_audio.mp3'):
    """유튜브 영상에서 오디오 추출 및 메타데이터 반환"""
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_path)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path.replace('.mp3', '')
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Unknown Title')
            channel = info.get('uploader', 'Unknown Artist')
            
            ydl.download([url])
            
        if not os.path.exists(output_path):
            raise Exception(f"Audio file was not created at {output_path}")
            
        return title, channel, output_path
    except Exception as e:
        print(f"다운로드 중 오류 발생: {str(e)}")
        raise

def get_harmonic_peaks(y, sr, frame_length=4096, hop_length=512, threshold=0.1):
    """하모닉 피크 검출"""
    # STFT 계산
    D = librosa.stft(y, n_fft=frame_length, hop_length=hop_length, window='blackman')
    S = np.abs(D)
    
    # 주파수 빈 계산
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
    
    # 피크 검출
    peaks = []
    for frame_idx in range(S.shape[1]):
        frame_peaks = []
        spectrum = S[:, frame_idx]
        
        # 스펙트럼이 너무 약한 경우 건너뛰기
        if np.max(spectrum) < 1e-6:
            peaks.append(frame_peaks)
            continue
            
        peak_idxs = librosa.util.peak_pick(spectrum, 
                                         pre_max=30, 
                                         post_max=30, 
                                         pre_avg=30, 
                                         post_avg=30, 
                                         delta=threshold, 
                                         wait=20)
        
        max_magnitude = np.max(spectrum)
        for idx in peak_idxs:
            if spectrum[idx] > threshold * max_magnitude:
                frame_peaks.append((freqs[idx], spectrum[idx]))
        
        # 프레임의 피크들을 크기순으로 정렬
        frame_peaks.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 5개의 피크만 유지
        frame_peaks = frame_peaks[:5]
        peaks.append(frame_peaks)
    
    return peaks, hop_length/sr

def calculate_note_duration(duration, tempo):
    """음표 길이를 계산하여 적절한 음표 타입 반환"""
    beat_duration = 60.0 / tempo  # 한 박자의 길이 (초)
    relative_duration = duration / beat_duration  # 박자 단위로 변환
    
    # 음표 길이 정의 (4분음표 기준)
    durations = {
        4.0: 'whole',      # 온음표
        3.0: 'half.',      # 점2분음표
        2.0: 'half',       # 2분음표
        1.5: 'quarter.',   # 점4분음표
        1.0: 'quarter',    # 4분음표
        0.75: 'eighth.',   # 점8분음표
        0.5: 'eighth',     # 8분음표
        0.375: '16th.',    # 점16분음표
        0.25: '16th',      # 16분음표
        0.1875: '32nd.',   # 점32분음표
        0.125: '32nd',     # 32분음표
        0.0625: '64th'     # 64분음표
    }
    
    # 가장 가까운 음표 길이 찾기
    closest_duration = min(durations.keys(), key=lambda x: abs(x - relative_duration))
    return durations[closest_duration]

def analyze_audio(audio_path):
    """오디오 파일 분석하여 음표 정보 추출"""
    try:
        # 오디오 파일 로드 및 전처리
        y, sr = librosa.load(audio_path, sr=44100)  # 샘플링 레이트 고정
        if len(y) == 0:
            raise ValueError("오디오 파일이 비어있습니다.")

        # 실제 템포 추정
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # 볼륨 정규화 및 노이즈 제거
        y = librosa.util.normalize(y)
        y = librosa.effects.preemphasis(y)
        
        # 하모닉/타악기 성분 분리
        y_harmonic, y_percussive = librosa.effects.hpss(y, margin=4.0)
        
        # 피아노 음역대 필터링
        fmin = librosa.note_to_hz('A0')  # 27.5 Hz
        fmax = librosa.note_to_hz('C8')  # 4186 Hz
        
        # STFT 파라미터 최적화
        n_fft = 4096
        hop_length = 512
        win_length = 2048
        
        # 스펙트로그램 계산
        D = librosa.stft(y_harmonic, n_fft=n_fft, hop_length=hop_length, 
                        win_length=win_length, window='hann')
        S = np.abs(D)
        
        # 주파수 빈 계산
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        times = librosa.times_like(S, sr=sr, hop_length=hop_length)
        
        # 온셋 검출 개선
        onset_env = librosa.onset.onset_strength(
            y=y_percussive,
            sr=sr,
            hop_length=hop_length,
            aggregate=np.median,
            fmin=fmin,
            fmax=fmax,
            n_mels=128,  # mel 필터 수 조정
            htk=True
        )
        
        # 적응형 임계값으로 온셋 검출
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
            delta=0.5,
            wait=2,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=3,
            backtrack=True
        )
        
        if len(onset_frames) == 0:
            raise ValueError("음표를 찾을 수 없습니다.")
        
        # 음표 정보 추출
        notes = []
        prev_magnitude_threshold = 0.1
        
        for i, onset_frame in enumerate(onset_frames):
            # 현재 프레임의 스펙트럼
            current_spectrum = S[:, onset_frame]
            
            # 다음 온셋까지의 지속 시간 계산
            if i < len(onset_frames) - 1:
                duration = times[onset_frames[i + 1]] - times[onset_frame]
            else:
                # 마지막 음표는 이전 음표들의 평균 지속 시간 사용
                if len(onset_frames) > 1:
                    avg_duration = np.mean([
                        times[onset_frames[j+1]] - times[onset_frames[j]]
                        for j in range(len(onset_frames)-1)
                    ])
                    duration = avg_duration
                else:
                    duration = 0.5  # 기본값 (500ms)
            
            # 동적 임계값 계산
            magnitude_threshold = max(
                prev_magnitude_threshold * 0.8,  # 이전 임계값의 80%
                np.max(current_spectrum) * 0.1   # 현재 스펙트럼 최대값의 10%
            )
            
            # 피크 검출
            peaks = librosa.util.peak_pick(
                current_spectrum,
                pre_max=30,
                post_max=30,
                pre_avg=30,
                post_avg=30,
                delta=magnitude_threshold,
                wait=20
            )
            
            # 피크를 MIDI 노트로 변환
            notes_at_onset = []
            for peak in peaks:
                freq = freqs[peak]
                if fmin <= freq <= fmax:
                    magnitude = current_spectrum[peak]
                    if magnitude > magnitude_threshold:
                        midi_note = int(round(librosa.hz_to_midi(freq)))
                        if 21 <= midi_note <= 108:  # 피아노 음역 체크
                            # 하모닉 검사
                            is_harmonic = False
                            for existing_note in notes_at_onset:
                                ratio = max(freq, existing_note['freq']) / min(freq, existing_note['freq'])
                                if abs(ratio - round(ratio)) < 0.1:  # 하모닉 비율 체크
                                    is_harmonic = True
                                    break
                            
                            if not is_harmonic:
                                note_info = {
                                    'note': midi_note,
                                    'freq': freq,
                                    'magnitude': float(magnitude),
                                    'time': float(times[onset_frame]),
                                    'duration': float(duration),
                                    'note_type': calculate_note_duration(duration, tempo)
                                }
                                notes_at_onset.append(note_info)
            
            # 화음 정리
            if len(notes_at_onset) > 1:
                # 화음 내 음표들 정렬 (낮은 음부터)
                notes_at_onset.sort(key=lambda x: x['note'])
                # 너무 가까운 음은 제거
                filtered_notes = []
                for j, note in enumerate(notes_at_onset):
                    if j == 0 or note['note'] - filtered_notes[-1]['note'] >= 2:
                        filtered_notes.append(note)
                if len(filtered_notes) > 1:
                    notes.append(filtered_notes)
                elif filtered_notes:
                    notes.append(filtered_notes[0])
            elif notes_at_onset:
                notes.append(notes_at_onset[0])
            
            # 다음 프레임을 위해 임계값 업데이트
            if notes_at_onset:
                prev_magnitude_threshold = max(n['magnitude'] for n in notes_at_onset)
        
        if not notes:
            raise ValueError("유효한 음표를 찾을 수 없습니다.")
        
        return notes, tempo
        
    except Exception as e:
        print(f"오디오 분석 중 오류 발생: {str(e)}")
        raise

def create_sheet_music(notes, tempo, output_file='sheet_music.pdf'):
    """
    음표 정보를 바탕으로 악보 생성
    """
    try:
        # 새로운 Score 객체 생성
        score = music21.stream.Score()
        
        # 피아노 파트 생성
        piano_part = music21.stream.Part()
        piano_part.insert(0, music21.instrument.Piano())
        
        # 메타데이터 설정
        score.insert(0, music21.metadata.Metadata())
        score.metadata.title = 'Piano Sheet Music'
        
        # 높은음자리표와 낮은음자리표 스태프 생성
        treble_staff = music21.stream.Part()
        bass_staff = music21.stream.Part()
        
        # 음자리표 설정
        treble_staff.insert(0, music21.clef.TrebleClef())
        bass_staff.insert(0, music21.clef.BassClef())
        
        # 박자표 설정
        time_signature = music21.meter.TimeSignature('4/4')
        treble_staff.insert(0, time_signature)
        bass_staff.insert(0, time_signature)
        
        # 템포 설정
        mm = music21.tempo.MetronomeMark(number=int(round(tempo)))
        treble_staff.insert(0, mm)
        
        # 첫 마디 생성
        current_measure_treble = music21.stream.Measure(number=1)
        current_measure_bass = music21.stream.Measure(number=1)
        
        # 현재 마디의 시간 추적
        current_time = 0.0
        measure_duration = 4.0  # 4/4 박자
        measure_count = 1
        
        def finish_current_measures():
            nonlocal current_measure_treble, current_measure_bass, current_time, measure_count
            # 빈 마디에 쉼표 추가
            if not current_measure_treble.notes and not current_measure_treble.notesAndRests:
                current_measure_treble.insert(0, music21.note.Rest(quarterLength=4.0))
            if not current_measure_bass.notes and not current_measure_bass.notesAndRests:
                current_measure_bass.insert(0, music21.note.Rest(quarterLength=4.0))
            
            # 마디 추가
            treble_staff.append(current_measure_treble)
            bass_staff.append(current_measure_bass)
            
            # 새 마디 준비
            measure_count += 1
            current_measure_treble = music21.stream.Measure(number=measure_count)
            current_measure_bass = music21.stream.Measure(number=measure_count)
            current_time = 0.0
        
        # 음표 처리
        for note_data in notes:
            if isinstance(note_data, list):  # 화음 처리
                treble_notes = [n for n in note_data if n['note'] >= 60]
                bass_notes = [n for n in note_data if n['note'] < 60]
                
                if treble_notes:
                    chord = create_chord(treble_notes)
                    if current_time + chord.duration.quarterLength > measure_duration:
                        finish_current_measures()
                    current_measure_treble.insert(current_time, chord)
                    current_time += chord.duration.quarterLength
                
                if bass_notes:
                    chord = create_chord(bass_notes)
                    current_measure_bass.insert(current_time, chord)
            
            else:  # 단일 음표 처리
                note = create_note(note_data)
                if current_time + note.duration.quarterLength > measure_duration:
                    finish_current_measures()
                
                if note_data['note'] >= 60:
                    current_measure_treble.insert(current_time, note)
                else:
                    current_measure_bass.insert(current_time, note)
                current_time += note.duration.quarterLength
        
        # 마지막 마디 처리
        if current_time > 0:
            finish_current_measures()
        
        # 빈 악보 처리
        if not treble_staff.getElementsByClass('Measure'):
            empty_measure = music21.stream.Measure(number=1)
            empty_measure.insert(0, music21.note.Rest(quarterLength=4.0))
            treble_staff.append(empty_measure)
            bass_staff.append(empty_measure)
        
        # 스트림 구조 구성
        piano_part.append(treble_staff)
        piano_part.append(bass_staff)
        score.append(piano_part)
        
        # MusicXML 파일로 저장
        xml_path = output_file.replace('.pdf', '.xml')
        score.write('musicxml', fp=xml_path)
        
        # MuseScore로 PDF 변환
        import subprocess
        musescore_path = "C:/Program Files/MuseScore 4/bin/MuseScore4.exe"
        
        try:
            subprocess.run([
                musescore_path,
                "-o", output_file,
                xml_path
            ], check=True)
            
            # MIDI 파일 생성
            midi_file = output_file.replace('.pdf', '.mid')
            score.write('midi', fp=midi_file)
            
            return midi_file
            
        except subprocess.CalledProcessError as e:
            print(f"MuseScore 변환 중 오류 발생: {str(e)}")
            return xml_path
            
    except Exception as e:
        print(f"악보 생성 중 오류 발생: {str(e)}")
        raise

def create_note(note_data):
    """단일 음표 생성"""
    pitch = music21.pitch.Pitch(midi=note_data['note'])
    note = music21.note.Note(pitch)
    note.duration = music21.duration.Duration(type=note_data['note_type'])
    note.volume.velocity = int(min(127, note_data['magnitude'] * 100))
    return note

def create_chord(notes_data):
    """화음 생성"""
    chord_notes = []
    note_type = notes_data[0]['note_type']
    
    for note in notes_data:
        pitch = music21.pitch.Pitch(midi=note['note'])
        n = music21.note.Note(pitch)
        n.duration = music21.duration.Duration(type=note_type)
        n.volume.velocity = int(min(127, note['magnitude'] * 100))
        chord_notes.append(n)
    
    return music21.chord.Chord(chord_notes)

def distribute_notes_between_hands(notes):
    """음표들을 양손으로 더 균형있게 분배"""
    if isinstance(notes, dict):
        notes = [notes]
    
    right_notes = []
    left_notes = []
    
    # 화음의 모든 음표 정렬 (pitch 기준)
    sorted_notes = sorted(notes, key=lambda x: x['note'])
    
    if len(sorted_notes) == 1:
        # 단일 음표의 경우, 이전 상태를 고려하여 분배
        note = sorted_notes[0]
        if 48 <= note['note'] <= 72:  # 더 넓은 중간 영역
            # 중간 영역은 음높이에 따라 비율적으로 분배
            if note['note'] >= 60:
                right_notes.append(note)
            else:
                left_notes.append(note)
        else:
            # 극단적인 높이/낮이는 해당 손으로
            if note['note'] > 72:
                right_notes.append(note)
            else:
                left_notes.append(note)
    else:
        # 화음의 경우
        total_notes = len(sorted_notes)
        
        if total_notes == 2:
            # 2개 음표는 간격에 따라 분배
            if sorted_notes[1]['note'] - sorted_notes[0]['note'] >= 12:  # 옥타브 이상
                left_notes.append(sorted_notes[0])
                right_notes.append(sorted_notes[1])
            else:
                # 가운데 C를 기준으로 분배
                mid_point = sum(n['note'] for n in sorted_notes) / len(sorted_notes)
                if mid_point >= 60:
                    right_notes.extend(sorted_notes)
                else:
                    left_notes.extend(sorted_notes)
        else:
            # 3개 이상의 음표는 더 균형있게 분배
            # 전체 음역대 계산
            total_range = sorted_notes[-1]['note'] - sorted_notes[0]['note']
            
            if total_range <= 12:  # 한 옥타브 이내
                # 평균 음높이로 분배
                avg_pitch = sum(n['note'] for n in sorted_notes) / len(sorted_notes)
                if avg_pitch >= 60:
                    right_notes.extend(sorted_notes)
                else:
                    left_notes.extend(sorted_notes)
            else:
                # 음표 수에 따라 분배 비율 조정
                split_index = total_notes // 2
                if total_notes >= 4:
                    # 4개 이상인 경우 더 균등하게 분배
                    split_index = (total_notes + 1) // 2
                
                # 간격이 너무 좁은 경우 조정
                while split_index > 0 and split_index < total_notes:
                    if sorted_notes[split_index]['note'] - sorted_notes[split_index-1]['note'] >= 5:
                        break
                    split_index += 1
                
                left_notes.extend(sorted_notes[:split_index])
                right_notes.extend(sorted_notes[split_index:])
    
    # 한 손에 너무 많은 음표가 몰린 경우 재조정
    max_notes_per_hand = 4
    if len(right_notes) > max_notes_per_hand and len(left_notes) < max_notes_per_hand:
        # 오른손 음표 일부를 왼손으로 이동
        notes_to_move = right_notes[:len(right_notes) - max_notes_per_hand]
        right_notes = right_notes[len(right_notes) - max_notes_per_hand:]
        left_notes.extend(notes_to_move)
    elif len(left_notes) > max_notes_per_hand and len(right_notes) < max_notes_per_hand:
        # 왼손 음표 일부를 오른손으로 이동
        notes_to_move = left_notes[len(left_notes) - max_notes_per_hand:]
        left_notes = left_notes[:len(left_notes) - max_notes_per_hand]
        right_notes.extend(notes_to_move)
    
    return right_notes, left_notes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                # 파일 저장
                audio_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                file.save(audio_path)
        elif 'youtube_url' in request.form:
            youtube_url = request.form['youtube_url']
            if youtube_url:
                # YouTube 동영상 다운로드
                title, channel, audio_path = download_youtube_audio(youtube_url)
        else:
            return jsonify({'success': False, 'error': '파일이나 YouTube URL을 제공해주세요.'})
        
        # 음악 분석 및 악보 생성
        notes, tempo = analyze_audio(audio_path)
        midi_file = create_sheet_music(
            notes,
            tempo,
            output_file=os.path.join(app.config['UPLOAD_FOLDER'], 'sheet_music.pdf')
        )
        
        return jsonify({
            'success': True,
            'sheet_music': 'sheet_music.pdf',
            'midi_file': midi_file
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/sheet_music/<filename>')
def get_sheet_music(filename):
    try:
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename),
                        mimetype='application/pdf')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/play_midi/<path:filename>')
def play_midi(filename):
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    app.run(debug=True)