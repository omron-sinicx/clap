import React from 'react';
import { render } from 'react-dom';

import Header from '@/components/header.jsx';
import Overview from '@/components/overview.jsx';
import Video from '@/components/video.jsx';
import Body from '@/components/body.jsx';
import Contact from '@/components/contact.jsx';
import Footer from '@/components/footer.jsx';
import Citation from '@/components/citation.jsx';
import SpeakerDeck from '@/components/speakerdeck.jsx';
import Projects from '@/components/projects.jsx';
import data from '../../template.yaml';

import '@/js/styles.js';

class Template extends React.Component {
  render() {
    return (
      <div>
        <Header
          title={data.title}
          conference={data.conference}
          authors={data.authors}
          affiliations={data.affiliations}
          meta={data.meta}
          resources={data.resources}
          theme={data.theme}
          header={data.header}
        />
        <div className="uk-container uk-container-small">
          <Overview
            overview={data.overview}
            teaser={data.teaser}
            description={data.description}
          />
          <Video video={data.resources.video} />
          <SpeakerDeck dataId={data.speakerdeck} />
          <Body body={data.body} />
          <Contact
            authors={data.authors}
            contact_ids={data.contact_ids}
            resources={data.resources}
          />
          <Citation bibtex={data.bibtex} />
          <Projects projects={data.projects} />
        </div>
        <Footer />
      </div>
    );
  }
}

render(<Template />, document.getElementById('root'));
