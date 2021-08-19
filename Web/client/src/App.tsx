import { FC, Fragment, useEffect, useState } from 'react';
import { io } from 'socket.io-client';
import { Line } from 'react-chartjs-2';
import { useRef } from 'react';
import { ChartConfiguration, ChartData, ChartOptions } from 'chart.js';
import styled from '@emotion/styled';

// This is all just testing, will move to other files later

const socket = io(process.env.NODE_ENV === 'development' ? 'http://localhost:5000' : '');

const Wrapper = styled.div`
	position: relative;
	width: 600px;
`;

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

const App: FC = () => {
	// Chart reference
	const ref = useRef<ChartConfiguration | any>(null);

	// Image state
	const [base64Image, setBase64Image] = useState('');

	// Subscribe to socket
	useEffect(() => {
		socket.on('image', image => {
			setBase64Image(image);
		});

		// socket.on('message', message => {
		// 	// Update data
		// 	const data = ref.current?.data.datasets[0].data;
		// 	data?.shift();
		// 	data?.push(message);

		// 	// Update chart
		// 	ref.current.update();
		// });
	}, []);

	return (
		<Fragment>
			<img style={{ height: 200, width: 200 }} src={`data:image/png;base64, ${base64Image}`} alt="" />
			{/* <Wrapper>
				<Line data={chartData} ref={ref} options={chartOptions} />
			</Wrapper>
			<button onClick={() => socket.emit('play')}>Play</button>
			<button onClick={() => socket.emit('pause')}>Pause</button> */}
		</Fragment>
	);
};

export default App;
