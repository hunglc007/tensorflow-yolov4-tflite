import { ChartConfiguration, ChartData, ChartOptions } from 'chart.js/auto';
import React, { FC, useEffect, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import socket from '../../Services/SocketService';
import { CardTitle } from '../../Styles/Containers';
import * as Styles from './SensorPlot.styles';

const chartData: ChartData = {
	labels: Array.from(Array(200).keys()),
	datasets: [
		{
			label: 'Sensor value',
			data: Array.from(Array(200)),
			fill: true,
			cubicInterpolationMode: 'monotone',
			backgroundColor: 'rgb(255, 99, 132, 0.2)',
			borderColor: 'rgba(255, 99, 132)',
		},
	],
};

const chartOptions: ChartOptions = {
	scales: {
		yAxes: {
			suggestedMax: 1.5,
			suggestedMin: -1.5,
		},
	},
	elements: {
		point: {
			radius: 0,
		},
	},
	responsive: true,
	maintainAspectRatio: false,
};

const SensorPlot: FC = () => {
	// Chart reference
	const ref = useRef<ChartConfiguration | any>(null);

	// Subscribe to socket
	useEffect(() => {
		socket.on('message', message => {
			// Update data
			const data = ref.current?.data.datasets[0].data;
			data?.shift();
			data?.push(message);

			// Update chart
			ref.current.update();
		});
	}, []);

	return (
		<Styles.Container>
			<CardTitle>Sensor Plot</CardTitle>
			<Styles.ChartContainer>
				<Line ref={ref} data={chartData} options={chartOptions} />
			</Styles.ChartContainer>
		</Styles.Container>
	);
};

export default SensorPlot;
