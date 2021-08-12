// Register aliases
import 'module-alias/register';

import path from 'path';
import express from 'express';
import { createServer } from 'http';
import { Server, Socket } from 'socket.io';
import Emitter from 'Models/Emitter';

// ----------------------------------------------------------------------------------------
// Initialisation
// ----------------------------------------------------------------------------------------
const app = express(); // Init app
const server = createServer(app); // Create http server
const io = new Server(server, { cors: { origin: 'http://localhost:3000' } }); // Init sockets

// ----------------------------------------------------------------------------------------
// Setup middleware
// ----------------------------------------------------------------------------------------
app.use(express.static(path.join(__dirname, '../../client/build'))); // Serve static files

// ----------------------------------------------------------------------------------------
// Configure sockets
// ----------------------------------------------------------------------------------------
io.on('connection', (socket: Socket) => {
	console.log(`User connected from ${socket.handshake.address}`);

	const emitter = new Emitter(socket);

	socket.on('pause', () => {
		emitter.pause();
	});

	socket.on('play', () => {
		emitter.play();
	});

	socket.on('disconnect', () => {
		console.log(`User from ${socket.handshake.address} has disconnected`);
		emitter.pause();
	});
});

// ----------------------------------------------------------------------------------------
// Start server
// ----------------------------------------------------------------------------------------
// Define port
const port = 5000;
// Listen for new connections
server.listen(port, () => console.log('App is listening on port ' + port));
