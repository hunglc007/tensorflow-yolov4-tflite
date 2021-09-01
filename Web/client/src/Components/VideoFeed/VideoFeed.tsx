import React, { FC } from 'react';
import { CardTitle } from '../../Styles/Containers';
import * as Styles from './VideoFeed.styles';

const VideoFeed: FC = () => {
	return (
		<Styles.Container>
			<CardTitle>Video Feed</CardTitle>
			<Styles.VideoContainer src="https://static.scientificamerican.com/sciam/cache/file/1BACC933-1D88-4EB8-9B73C5B296F102F5_source.jpg" />
		</Styles.Container>
	);
};

export default VideoFeed;
