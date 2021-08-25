import { FC, Fragment, useEffect } from 'react';
import { io } from 'socket.io-client';
import { Line } from 'react-chartjs-2';
import { useRef } from 'react';
import { ChartConfiguration, ChartData, ChartOptions } from 'chart.js';
import styled from '@emotion/styled';
import Card from '@material-ui/core'

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
			suggestedMax: 1,
			suggestedMin: -1,
		},
	},
	elements: {
		point: {
			radius: 0,
		},
	},
};
export default function Chart() {
    return(
		<div>
			<Line data={chartData} options={chartOptions} />
			<h3>Real Time</h3>
			<Line data={chartData} options={chartOptions} />
			<h3>Total Data</h3>
		</div>
        
    )
}
