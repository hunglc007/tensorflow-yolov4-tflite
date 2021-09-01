import styled from '@emotion/styled';

export const AppContainer = styled.div`
	display: flex;
	overflow: hidden;
	height: 100vh;
	width: 100vw;
`;

export const Card = styled.div`
	display: flex;
	flex-direction: column;
	overflow: hidden;
	margin: 1rem;
	flex: 1;
	border: 1px solid #ededed;
	box-shadow: 2px 2px 10px #ededed;
	border-radius: 0.5rem;
`;

export const CardTitle = styled.div`
	padding: 0.5rem;
	border-bottom: 1px solid #ededed;
`;

export const IPContainer = styled.div`
	flex: 2;
	display: flex;
	flex-direction: column;
`;

export const AQSContainer = styled.div`
	flex: 2;
	display: flex;
	flex-direction: column;
`;
