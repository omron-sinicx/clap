import React from 'react';
import { FaRegFilePdf } from 'react-icons/fa';

// Poster thumbnails that open in a <dialog>. When the View Transition API is
// available, the thumbnail morphs into the enlarged poster (and back on close);
// otherwise the dialog just opens/closes without animation.
const VT_NAME = 'poster-zoom';

const withTransition = (update) => {
  const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)');
  if (!document.startViewTransition || reduce?.matches) {
    update();
    return Promise.resolve();
  }
  return document.startViewTransition(update).finished;
};

export default class Posters extends React.Component {
  constructor(props) {
    super(props);
    this.dialog = React.createRef();
    this.modalImg = React.createRef();
    this.thumbs = [];
    this.state = { active: null };
  }

  open(idx) {
    const thumb = this.thumbs[idx];
    const poster = this.props.posters[idx];
    const dialog = this.dialog.current;
    const img = this.modalImg.current;
    thumb.style.viewTransitionName = VT_NAME;
    withTransition(() => {
      // DOM is mutated synchronously so the "new" snapshot is correct.
      thumb.style.viewTransitionName = '';
      img.src = poster.image;
      img.alt = poster.title;
      img.style.viewTransitionName = VT_NAME;
      this.setState({ active: idx });
      dialog.showModal();
    });
  }

  close() {
    const idx = this.state.active;
    const dialog = this.dialog.current;
    if (idx === null || !dialog.open) return;
    const thumb = this.thumbs[idx];
    const img = this.modalImg.current;
    withTransition(() => {
      img.style.viewTransitionName = '';
      dialog.close();
      thumb.style.viewTransitionName = VT_NAME;
      this.setState({ active: null });
    }).finally(() => (thumb.style.viewTransitionName = ''));
  }

  render() {
    const { posters } = this.props;
    if (!posters) return null;
    const active =
      this.state.active !== null ? posters[this.state.active] : null;
    return (
      <div className="posters">
        <div
          className={`uk-grid uk-grid-small uk-child-width-1-${Math.min(posters.length, 2)}@s`}
          uk-grid=""
        >
          {posters.map((p, idx) => (
            <div key={p.pdf}>
              <button
                type="button"
                className="poster-thumb"
                onClick={() => this.open(idx)}
                aria-label={`Open poster: ${p.title}`}
              >
                <img
                  ref={(el) => (this.thumbs[idx] = el)}
                  src={p.image}
                  alt={p.title}
                  loading="lazy"
                />
              </button>
              <p className="uk-text-meta uk-text-center uk-margin-small-top">
                {p.title}
                {p.venue ? <> &middot; {p.venue}</> : null} &middot;{' '}
                <a href={p.pdf} target="_blank" rel="noreferrer">
                  PDF
                </a>
              </p>
            </div>
          ))}
        </div>

        <dialog
          ref={this.dialog}
          className="poster-dialog"
          onClick={(e) => e.target === this.dialog.current && this.close()}
          onCancel={(e) => {
            e.preventDefault();
            this.close();
          }}
        >
          <div className="poster-dialog-bar">
            <span>{active?.title}</span>
            <span>
              {active && (
                <a
                  href={active.pdf}
                  target="_blank"
                  rel="noreferrer"
                  className="uk-margin-small-right"
                >
                  <FaRegFilePdf /> Open PDF
                </a>
              )}
              <button
                type="button"
                className="uk-close-large"
                uk-close=""
                aria-label="Close"
                onClick={() => this.close()}
              />
            </span>
          </div>
          <a href={active?.pdf} target="_blank" rel="noreferrer">
            <img ref={this.modalImg} alt="" />
          </a>
        </dialog>
      </div>
    );
  }
}
