import { Global } from '@emotion/react';
import React, { FC } from 'react';
import GlobalStyles from './Styles/Global';
import * as Styles from './Styles/Containers';
import ImageStream from './Components/ImageStream/ImageStream';
import VideoFeed from './Components/VideoFeed/VideoFeed';
import TargetFeed from './Components/TargetFeed/TargetFeed';
import SensorPlot from './Components/SensorPlot/SensorPlot';
import SensorFeed from './Components/SensorFeed/SensorFeed';
import { StylesProvider } from '@material-ui/core';

const App: FC = () => {
	return (
		<StylesProvider injectFirst>
			<Styles.AppContainer>
				<Global styles={GlobalStyles} />

				<ImageStream />
				<Styles.IPContainer>
					<VideoFeed />
					<TargetFeed />
				</Styles.IPContainer>
				<Styles.AQSContainer>
					<SensorPlot />
					<SensorFeed />
				</Styles.AQSContainer>
			</Styles.AppContainer>
		</StylesProvider>
	);
};

export default App;

// import { FC, Fragment, useEffect, useState } from 'react';
// import { io } from 'socket.io-client';
// import { Line } from 'react-chartjs-2';
// import { useRef } from 'react';
// import { ChartConfiguration, ChartData, ChartOptions } from 'chart.js';
// import styled from '@emotion/styled';
// import Navbar from './Components/Navbar';
// import PhotoSlider from './Components/PhotoSlider';
// import Chart from './Components/Chart';
// import Card from '@material-ui/core/Card';
// import { CardMedia } from '@material-ui/core';

// // This is all just testing, will move to other files later

// const socket = io(process.env.NODE_ENV === 'development' ? 'http://localhost:5000' : '');

// const Wrapper = styled.div`
// 	position: relative;
// 	width: 600px;
// `;

// const chartData: ChartData = {
// 	labels: Array.from(Array(200).keys()),
// 	datasets: [
// 		{
// 			label: 'Sensor value',
// 			data: Array.from(Array(200)),
// 			fill: true,
// 			cubicInterpolationMode: 'monotone',
// 			backgroundColor: 'rgb(255, 99, 132, 0.2)',
// 			borderColor: 'rgba(255, 99, 132)',
// 		},
// 	],
// };

// const chartOptions: ChartOptions = {
// 	scales: {
// 		yAxes: {
// 			suggestedMax: 1,
// 			suggestedMin: -1,
// 		},
// 	},
// 	elements: {
// 		point: {
// 			radius: 0,
// 		},
// 	},
// };

// const App: FC = () => {
// 	// Chart reference
// 	const ref = useRef<ChartConfiguration | any>(null);

// 	// Image state
// 	const [base64Image, setBase64Image] = useState('');

// 	// Subscribe to socket
// 	useEffect(() => {
// 		socket.on('image', image => {
// 			setBase64Image(image);
// 		});

// 		socket.on('message', message => {
// 		 	// Update data
// 		const data = ref.current?.data.datasets[0].data;
// 		data?.shift();
// 		 	data?.push(message);

// 			// Update chart
// 		 	ref.current.update();
// 		});
// 	}, []);

// 	return (
// 		<div>
// 			<Navbar/>
// 			<PhotoSlider/>
// 			<div style={{display: 'inline-flex', padding: 50}}>
// 				<Card>
// 					<img style={{ height: 600, width: 800 }}
// 					src='https://i.pinimg.com/originals/62/7b/80/627b80d98af7bbc993bf61115418f71d.jpg'
// 					alt='Drone Stream'
// 					/>
// 					<div style={{flex: 'inline-flex'}} >
// 						<button onClick={() => socket.emit('play')}>Play</button>
// 						<button onClick={() => socket.emit('pause')}>Pause</button>
// 						<h2>Live Camera Feed</h2>
// 					</div>

// 				</Card>
// 				<Fragment>
// 					<img style={{ height: 400, width: 200 }} src={`data:image/png;base64, ${base64Image}`} alt="" />
// 						<Wrapper>
// 							<Chart/>
// 							<div style={{flex: 'inline-flex'}} >
// 								<button onClick={() => socket.emit('play')}>Play</button>
// 								<button onClick={() => socket.emit('pause')}>Pause</button>
// 							</div>
// 						</Wrapper>

// 				</Fragment>

// 			</div>
// 		</div>

// 		//<Fragment>
// 		//	<img style={{ height: 200, width: 200 }} src={`data:image/png;base64, ${base64Image}`} alt="" />
// 			/* <Wrapper>
// 				<Line data={chartData} ref={ref} options={chartOptions} />
// 			</Wrapper>
// 			<button onClick={() => socket.emit('play')}>Play</button>
// 			<button onClick={() => socket.emit('pause')}>Pause</button> */
// 		//</Fragment>
// 	);
// };

// export default App;
