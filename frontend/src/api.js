const BACKEND = "http://localhost:8000";

export async function initSegmentation(file, prompt = "cow") {
  const form = new FormData();
  form.append("image", file);
  form.append("meta", new Blob([JSON.stringify({ prompt })], { type: "application/json" }));

  const res = await fetch(`${BACKEND}/segment/init`, {
    method: "POST",
    body: form
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function refineMask({ image_id, instance_id, points }) {
  const res = await fetch(`${BACKEND}/segment/refine`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_id, instance_id, points })
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function maskPngFromB64(b64) {
  const byteCharacters = atob(b64);
  const byteNumbers = new Array(byteCharacters.length);
  for (let i = 0; i < byteCharacters.length; i++) byteNumbers[i] = byteCharacters.charCodeAt(i);
  const blob = new Blob([new Uint8Array(byteNumbers)], { type: "image/png" });
  return URL.createObjectURL(blob);
}
