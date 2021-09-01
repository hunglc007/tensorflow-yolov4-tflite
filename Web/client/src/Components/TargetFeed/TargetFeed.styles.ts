import styled from '@emotion/styled';
import { TableContainer } from '@material-ui/core';
import { Card } from '../../Styles/Containers';

export const Container = styled(Card)`
	margin: 1rem;
	margin-right: 0rem;
	flex: 1.5;
`;

export const Table = styled(TableContainer)`
	th,
	td {
		font-family: 'Poppins', sans-serif;
	}
`;
