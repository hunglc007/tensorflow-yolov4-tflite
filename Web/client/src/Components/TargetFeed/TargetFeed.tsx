import { TableBody, TableCell, TableContainer, TableHead, TableRow } from '@material-ui/core';
import React, { FC } from 'react';
import { CardTitle } from '../../Styles/Containers';
import * as Styles from './TargetFeed.styles';

const TargetFeed: FC = () => {
	return (
		<Styles.Container>
			<CardTitle>Target Feed</CardTitle>
			<Styles.Table>
				<TableHead>
					<TableRow>
						<TableCell width="80%">Message</TableCell>
						<TableCell width="20%">Timestamp</TableCell>
						<TableCell />
					</TableRow>
				</TableHead>
				<TableBody>
					<TableRow>
						<TableCell>Target B detected</TableCell>
						<TableCell>{new Date().toLocaleTimeString()}</TableCell>
						<TableCell />
					</TableRow>
					<TableRow>
						<TableCell>Target A detected</TableCell>
						<TableCell>{new Date().toLocaleTimeString()}</TableCell>
						<TableCell />
					</TableRow>
					<TableRow>
						<TableCell>Target A detected</TableCell>
						<TableCell>{new Date().toLocaleTimeString()}</TableCell>
						<TableCell />
					</TableRow>
				</TableBody>
			</Styles.Table>
		</Styles.Container>
	);
};

export default TargetFeed;
