import styled from '@emotion/styled';
import { Card } from '../../Styles/Containers';

export const Container = styled(Card)`
	flex: 1;
	margin: 1rem;
	margin-right: 0rem;
`;

export const ImageContainer = styled.div`
	height: 100%;
	overflow: auto;
	display: flex;
	flex-direction: column;
`;

export const Image = styled.img`
	padding: 0.5rem;
	height: 100%;
	width: 100%;
	object-fit: contain;
`;
