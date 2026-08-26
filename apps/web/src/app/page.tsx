export default function Home() {
  return (
    <div className="panel stack">
      <h1>Hear one speaker at a time</h1>
      <p className="muted">
        Upload a recording with overlapping speech, pick a speaker, and hear only them — in sync
        with the video, with their own captions.
      </p>
      <p>
        <a href="/login">
          <button type="button">Get started</button>
        </a>
      </p>
    </div>
  );
}
