import React from 'react';
import { FaBuildingColumns, FaBuilding } from 'react-icons/fa6';

export default class Authors extends React.Component {
  render() {
    const numAuthors = this.props.authors.length;
    const numAff = this.props.affiliations.length;
    const perRow = numAuthors > 4 ? Math.ceil(numAuthors / 2) : numAuthors;
    const authorClass = `uk-width-1-2 uk-width-1-${perRow}@s`;
    const affiliationClass = `uk-width-1-1 uk-width-1-${numAff}@s uk-margin-small-top`;
    const hasEqual = this.props.authors.some((a) => a.equal);
    const hasIntern = this.props.authors.some((a) => a.intern);

    return (
      <div>
        <div
          className="uk-text-primary uk-text-center uk-flex-center uk-grid-column-collapse uk-grid-row-small"
          data-uk-grid
        >
          {this.props.authors.map((author, idx) => {
            return (
              <span className={authorClass} key={'author-' + idx}>
                <a
                  target="_blank"
                  rel="noopener noreferrer"
                  className="uk-link-toggle"
                  href={author.url}
                >
                  {author.name}
                </a>
                <sup>
                  {author.affiliation.join(',')}
                  {author.equal ? '†' : ''}
                  {author.intern ? '‡' : ''}
                </sup>
              </span>
            );
          })}
        </div>
        <div
          className="uk-text-primary uk-text-center uk-flex-center uk-grid-column-collapse uk-grid-row-small"
          data-uk-grid
        >
          {this.props.affiliations.map((affiliation, idx) => {
            return (
              <span className={affiliationClass} key={'affiliation-' + idx}>
                <sup>{idx + 1}</sup>
                {affiliation.toLowerCase().includes('university') ? (
                  <FaBuildingColumns className="uk-margin-small-right" />
                ) : affiliation.toLowerCase().includes('omron sinic x') ? (
                  <FaBuilding className="uk-margin-small-right" />
                ) : null}
                {affiliation}
              </span>
            );
          })}
          {hasEqual && (
            <span className="uk-width-1-1 uk-text-small uk-margin-small-top">
              <sup>†</sup>Equal contribution
            </span>
          )}
          {hasIntern && (
            <span className="uk-width-1-1 uk-text-small">
              <sup>‡</sup>Work done as an internship project at OMRON SINIC X.
            </span>
          )}
          <span className="uk-width-1-1">{this.props.meta}</span>
        </div>
      </div>
    );
  }
}
