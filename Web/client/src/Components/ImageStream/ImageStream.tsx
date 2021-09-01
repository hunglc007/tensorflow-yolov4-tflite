import React, { FC } from 'react';
import { Card, CardTitle } from '../../Styles/Containers';
import * as Styles from './ImageStream.styles';

const ImageStream: FC = () => {
	return (
		<Styles.Container>
			<CardTitle>Image Stream</CardTitle>
			<Styles.ImageContainer>
				<Styles.Image src="https://static.scientificamerican.com/sciam/cache/file/1BACC933-1D88-4EB8-9B73C5B296F102F5_source.jpg" />
				<Styles.Image src="https://static.scientificamerican.com/sciam/cache/file/1BACC933-1D88-4EB8-9B73C5B296F102F5_source.jpg" />
				<Styles.Image src="https://static.scientificamerican.com/sciam/cache/file/1BACC933-1D88-4EB8-9B73C5B296F102F5_source.jpg" />
				<Styles.Image src="https://static.scientificamerican.com/sciam/cache/file/1BACC933-1D88-4EB8-9B73C5B296F102F5_source.jpg" />
				<Styles.Image src="https://static.scientificamerican.com/sciam/cache/file/1BACC933-1D88-4EB8-9B73C5B296F102F5_source.jpg" />
				<Styles.Image src="https://static.scientificamerican.com/sciam/cache/file/1BACC933-1D88-4EB8-9B73C5B296F102F5_source.jpg" />
			</Styles.ImageContainer>
		</Styles.Container>
	);
};

export default ImageStream;
