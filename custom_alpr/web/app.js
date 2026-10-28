'use strict';
const $ = id => document.getElementById(id);
let selectedFile = null, previewURL = null, currentJob = null, page = 0, total = 0, activeItem = null, currentView = 'plates';
const fileInput = $('file'), mediaArea = $('media-area');
function error(message) { $('error').textContent = message; $('error').hidden = !message; }
function selectFile(file) {
  if (!file) return;
  selectedFile = null;
  if (!/\.(jpe?g|png|webp|mp4)$/i.test(file.name)) return error('Hãy chọn JPG, PNG, WebP hoặc MP4.');
  if (file.size > 300 * 1024 * 1024) return error('Tệp vượt quá giới hạn 300 MB.');
  selectedFile = file; error('');
  currentJob = null;
  history.replaceState(null, '', '/');
  $('gallery').replaceChildren(); $('empty-gallery').hidden = false;
  $('empty-gallery').textContent = 'Bấm Nhận diện biển số để tạo kết quả cho tệp vừa chọn.';
  $('count').textContent = '0'; $('pagination').hidden = true;
  $('download-csv').hidden = true; $('process').hidden = true; $('live-inspector').hidden = true; $('live-badge').hidden = true; $('live-metrics').hidden = true;
  $('file-info').hidden = false; $('file-info').textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = URL.createObjectURL(file);
  showMedia(previewURL, /\.mp4$/i.test(file.name) ? 'video' : 'image');
  $('media-caption').textContent = file.name;
  $('summary').textContent = 'Bản gốc · chưa nhận diện';
  $('download-media').hidden = true;
}
function showMedia(url, kind) {
  const media = document.createElement(kind === 'video' ? 'video' : 'img');
  media.src = url;
  if (kind === 'video') { media.controls = true; media.playsInline = true; media.preload = 'metadata'; }
  else media.alt = 'Ảnh tải lên hoặc kết quả nhận diện';
  mediaArea.replaceChildren(media);
}
fileInput.addEventListener('change', () => selectFile(fileInput.files[0]));
$('confidence').addEventListener('input', e => $('conf-value').textContent = Number(e.target.value).toFixed(2));
for (const event of ['dragenter', 'dragover']) $('dropzone').addEventListener(event, e => { e.preventDefault(); $('dropzone').classList.add('dragover'); });
for (const event of ['dragleave', 'drop']) $('dropzone').addEventListener(event, e => { e.preventDefault(); $('dropzone').classList.remove('dragover'); });
$('dropzone').addEventListener('drop', e => { if (!fileInput.disabled) { fileInput.required = false; selectFile(e.dataTransfer.files[0]); } });
function busy(state) { for (const id of ['submit','sample','file','confidence','denoise','sharpen','padding','rectify','image-size','result-view','selection']) $(id).disabled = state; }
async function jsonFetch(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw Error(data.error || `Yêu cầu thất bại (${response.status}).`);
  return data;
}
async function start(url, body) {
  busy(true); error(''); $('process').hidden = false; $('phase').textContent = 'Đang tải lên'; $('percent').textContent = ''; $('progress').removeAttribute('value'); $('progress-detail').textContent = 'Giữ trang mở trong lúc xử lý.';
  $('gallery').replaceChildren(); $('empty-gallery').hidden = false; $('empty-gallery').textContent = 'Đang chuẩn bị kết quả đối chiếu…'; $('count').textContent = '0'; $('pagination').hidden = true; $('download-csv').hidden = true; $('download-media').hidden = true; $('live-inspector').hidden = true; $('live-badge').hidden = true; $('live-metrics').hidden = true;
  try {
    const {id} = await jsonFetch(url, {method: 'POST', body});
    currentJob = id;
    history.replaceState(null, '', `/?job=${id}`);
    await poll(id);
  } catch (e) { error(e.message); $('phase').textContent = 'Chưa hoàn tất'; $('progress').value = 0; $('empty-gallery').textContent = 'Chưa có kết quả. Hãy kiểm tra tệp và thử lại.'; }
  finally { busy(false); $('live-badge').hidden = true; }
}
function processingForm() {
  const form = new FormData();
  form.append('confidence', $('confidence').value);
  form.append('image_size', $('image-size').value);
  form.append('selection', $('selection').value);
  for (const name of ['denoise', 'sharpen', 'padding', 'rectify']) form.append(name, $(name).checked ? '1' : '0');
  return form;
}
function ocrLabel(item) { return item.mean_ocr_confidence == null ? 'OCR' : 'OCR TB'; }
function averageLabel(item) {
  return item.mean_ocr_confidence == null ? '' : ` · TB ${item.averaged_frames}/${item.window_frames} frame cùng chuỗi · Crop đại diện ${(item.representative_ocr_confidence*100).toFixed(1)}%`;
}
function processingLabel(options = {}) {
  const enabled = [];
  if (options.rectify) enabled.push('thử nắn thẳng');
  if (options.denoise) enabled.push('giảm nhiễu');
  if (options.sharpen) enabled.push('tăng nét');
  return enabled.length ? enabled.join(' + ') : 'không lọc ảnh';
}
$('upload-form').addEventListener('submit', e => { e.preventDefault(); if (!selectedFile) return error('Hãy chọn ảnh hoặc MP4.'); const form = processingForm(); form.append('file', selectedFile); start('/api/jobs', form); });
$('sample').addEventListener('click', () => start('/api/sample', processingForm()));
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function poll(id) {
  let streamAttached = false, settingsShown = false;
  while (true) {
    const job = await jsonFetch(`/api/jobs/${id}`);
    $('phase').textContent = job.phase;
    if (!settingsShown) {
      $('image-size').value = job.image_size || 640;
      $('selection').value = job.temporal_selection || 'average';
      $('confidence').value = job.confidence; $('conf-value').textContent = Number(job.confidence).toFixed(2);
      for (const name of ['denoise', 'sharpen', 'padding', 'rectify']) $(name).checked = Boolean(job.preprocessing?.[name]);
      settingsShown = true;
    }
    if (job.kind === 'video' && !streamAttached && !['done', 'error'].includes(job.status)) {
      showMedia(`/api/jobs/${id}/live`, 'image');
      $('media-caption').textContent = `${job.name} · frame đang được nhận diện`;
      $('summary').textContent = 'Đang chờ model…';
      $('live-badge').hidden = false; $('live-inspector').hidden = false; $('live-metrics').hidden = false;
      $('live-crop-pair').hidden = true;
      $('live-plate-text').textContent = 'Đang chờ frame đầu tiên…'; $('live-plate-meta').textContent = '';
      streamAttached = true;
    }
    if (job.kind === 'video' && job.processed_frames) {
      $('source-fps').textContent = Number(job.source_fps).toFixed(1);
      $('inference-fps').textContent = `${job.inference_fps} fps`;
      $('frame-ms').textContent = `${job.frame_ms} ms`;
      $('skipped-frames').textContent = job.skipped_frames;
      $('summary').textContent = `${timeLabel(job.time_seconds)} · frame ${job.latest_frame} · ${job.live_fps} FPS cập nhật`;
      $('live-crop-pair').hidden = !job.latest;
      if (job.latest) {
        const item = job.latest;
        $('live-crop-image').src = `/media/${id}/${item.crop}`;
        $('live-input-image').src = `/media/${id}/${item.ocr_input}`;
        $('live-plate-text').textContent = item.text || '(chuỗi trống)';
        $('live-plate-meta').textContent = `Frame ${item.frame} · ${timeLabel(item.time_seconds)} · ${item.width} × ${item.height} px · YOLO ${(item.detection_confidence*100).toFixed(1)}% · ${ocrLabel(item)} ${(item.ocr_confidence*100).toFixed(1)}%${averageLabel(item)} · ${item.padding ? "viền 5% · " : ""}${item.rectified ? "đã nắn thẳng · " : ""}${processingLabel(item.preprocessing)}`;
        $('live-crop').onclick = $('live-input').onclick = () => openCrop(item);
      } else {
        $('live-plate-text').textContent = 'Không có box qua ngưỡng';
        $('live-plate-meta').textContent = `Frame ${job.latest_frame} · ${timeLabel(job.time_seconds)}`;
      }
    }
    if (job.total_frames) { const pct = Math.min(99, Math.floor(100 * job.frames / job.total_frames)); $('progress').value = pct; $('percent').textContent = `${pct}%`; }
    $('progress-detail').textContent = `${job.processed_frames ?? job.frames} frame đã xử lý · ${job.detections} lượt OCR · ${job.plate_count ?? 0} biển đã gom${job.skipped_frames ? ` · bỏ qua ${job.skipped_frames} frame để giữ nhịp clip` : ''}`;
    if (job.status === 'error') throw Error(job.error);
    if (job.status === 'done') {
      $('progress').value = 100; $('percent').textContent = '100%'; $('live-badge').hidden = true; $('live-inspector').hidden = true;
      showMedia(`/media/${id}/${job.media}`, job.kind);
      $('media-caption').textContent = job.name;
      $('summary').textContent = `${job.frames} frame · ${job.seconds}s xử lý`;
      $('count').textContent = job.plate_count ?? 0;
      $('download-media').href = `/media/${id}/${job.media}`; $('download-media').hidden = false;
      $('download-csv').hidden = false;
      page = 0; await loadPage();
      return;
    }
    await pause(150);
  }
}
function timeLabel(seconds) { const m = Math.floor(seconds/60); return `${String(m).padStart(2,'0')}:${(seconds % 60).toFixed(2).padStart(5,'0')}`; }
function seek(item) { const video = mediaArea.querySelector('video'); if (!video) return; video.pause(); video.currentTime = item.time_seconds; mediaArea.scrollIntoView({behavior:'smooth', block:'center'}); }
function openCrop(item) {
  activeItem = item; $('zoom-image').src = `/media/${currentJob}/${item.crop}`;
  $('zoom-image').classList.remove('smooth'); $('zoom-input-image').classList.remove('smooth');
  $('smooth').textContent = 'Làm mượt hiển thị';
  $('processed-figure').hidden = !item.ocr_input;
  if (item.ocr_input) $('zoom-input-image').src = `/media/${currentJob}/${item.ocr_input}`;
  $('zoom-text').textContent = item.text || '(chuỗi trống)';
  $('zoom-meta').textContent = `Crop gốc ${item.width} × ${item.height} px · YOLO ${(item.detection_confidence*100).toFixed(1)}% · ${ocrLabel(item)} ${(item.ocr_confidence*100).toFixed(1)}%${averageLabel(item)} · Frame ${item.frame} · ${timeLabel(item.time_seconds)} · ${item.padding ? "viền 5% · " : ""}${item.rectified ? "đã nắn thẳng · " : ""}${processingLabel(item.preprocessing)}${item.observations ? ` · ${item.observations} lần quan sát · đồng thuận ${(item.text_agreement*100).toFixed(0)}%` : ''}`;
  $('seek').hidden = !mediaArea.querySelector('video'); $('crop-dialog').showModal();
}
async function loadPage() {
  const data = await jsonFetch(`/api/jobs/${currentJob}/results?page=${page}&view=${currentView}&selection=${$('selection').value}`);
  total = data.total; $('gallery').replaceChildren(); $('count').textContent = total;
  $('download-csv').href = currentView === 'plates' ? `/api/jobs/${currentJob}/results?format=csv&selection=${$('selection').value}` : `/media/${currentJob}/results.csv`;
  $('download-csv').textContent = currentView === 'plates' ? 'Tải CSV đã gom ↗' : 'Tải CSV từng frame ↗';
  $('empty-gallery').hidden = total !== 0;
  $('empty-gallery').textContent = 'Không tìm thấy biển số với ngưỡng hiện tại. Bạn có thể thử giảm ngưỡng YOLO hoặc dùng ảnh rõ hơn.';
  for (const item of data.items) {
    const card = document.createElement('article'); card.className = 'crop-card';
    const crop = document.createElement('button'); crop.className = 'crop-button'; crop.setAttribute('aria-label', `Phóng lớn biển ${item.text}`);
    const image = document.createElement('img'); image.loading = 'lazy'; image.src = `/media/${currentJob}/${item.crop}`; image.alt = `Crop biển số, OCR ${item.text}`;
    const expand = document.createElement('span'); expand.className = 'expand'; expand.textContent = '⤢'; crop.append(image, expand); crop.onclick = () => openCrop(item);
    const info = document.createElement('div'); info.className = 'crop-info';
    const text = document.createElement('p'); text.className = 'plate-text'; text.textContent = item.text || '(trống)';
    const scores = document.createElement('div'); scores.className = 'scores';
    for (const [label, score] of [['YOLO',item.detection_confidence], [ocrLabel(item),item.ocr_confidence]]) { const span = document.createElement('span'); span.textContent = `${label} `; const b = document.createElement('b'); b.textContent = `${(score*100).toFixed(1)}%`; span.append(b); scores.append(span); }
    const groupInfo = document.createElement('div'); groupInfo.className = 'observation-info';
    if (item.observations) {
      groupInfo.textContent = `${item.observations} lần · ${timeLabel(item.first_time)} – ${timeLabel(item.last_time)}`;
      const agreement = document.createElement('small');
      const stable = (item.window_frames || item.observations) >= 3 && item.text_agreement >= 0.6;
      agreement.textContent = `Đồng thuận ${(item.text_agreement*100).toFixed(0)}%${item.window_frames ? ` trong ${item.window_frames} frame gần nhất` : ''} · ${stable ? 'ổn định' : 'cần kiểm tra'}${averageLabel(item)}`;
      groupInfo.append(agreement);
    } else groupInfo.hidden = true;
    const meta = document.createElement('div'); meta.className = 'crop-meta'; const size = document.createElement('span'); size.textContent = `${item.width} × ${item.height} px`;
    const timestamp = document.createElement('button'); timestamp.className = 'time-button'; timestamp.textContent = mediaArea.querySelector('video') ? `▷ ${timeLabel(item.time_seconds)}` : 'Phóng lớn ⤢'; timestamp.onclick = () => mediaArea.querySelector('video') ? seek(item) : openCrop(item); meta.append(size,timestamp); info.append(text,scores,groupInfo,meta); card.append(crop,info); $('gallery').append(card);
  }
  const pages = Math.ceil(total/24); $('pagination').hidden = pages <= 1; $('page-label').textContent = `Trang ${page+1} / ${pages} · ${total} lượt phát hiện`; $('previous').disabled = page === 0; $('next').disabled = page+1 >= pages;
}
$('previous').onclick = async () => { if (page) { page--; try { await loadPage(); } catch(e) { error(e.message); } } };
$('next').onclick = async () => { if ((page+1)*24 < total) { page++; try { await loadPage(); } catch(e) { error(e.message); } } };
$('close-dialog').onclick = () => $('crop-dialog').close();
$('smooth').onclick = () => { const enabled = $('zoom-image').classList.toggle('smooth'); $('zoom-input-image').classList.toggle('smooth', enabled); $('smooth').textContent = enabled ? 'Tắt làm mượt hiển thị' : 'Làm mượt hiển thị'; };
$('seek').onclick = () => { $('crop-dialog').close(); seek(activeItem); };
$('crop-dialog').addEventListener('click', e => { if (e.target === $('crop-dialog')) { const rect = $('crop-dialog').getBoundingClientRect(); if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) $('crop-dialog').close(); } });

// A running clip can be reopened without restarting inference.
const resumedJob = new URLSearchParams(location.search).get('job');
if (resumedJob && /^[a-f0-9]{32}$/.test(resumedJob)) {
  currentJob = resumedJob; busy(true); $('process').hidden = false;
  poll(resumedJob).catch(e => error(e.message)).finally(() => { busy(false); $('live-badge').hidden = true; });
}

$('result-view').addEventListener('change', async e => {
  currentView = e.target.value; page = 0;
  if (currentJob) { try { await loadPage(); } catch (err) { error(err.message); } }
});

$('selection').addEventListener('change', async () => { page = 0; if (currentJob) { try { await loadPage(); } catch (err) { error(err.message); } } });
