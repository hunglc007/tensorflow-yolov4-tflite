import { Socket } from 'socket.io';
import fs from 'fs';
import path from 'path';
class Emitter {
	// Attributes
	interval: NodeJS.Timeout;
	socket: Socket;
	i = 1;

	// Init
	constructor(socket: Socket) {
		this.socket = socket;
		this.emit();
	}

	// Emit data
	private emit() {
		this.interval = setInterval(() => {
			const filePath =
				this.i % 2 === 0 ? path.join(__dirname, '../result.png') : path.join(__dirname, '../result2.jpeg');
			const imgFile = fs.readFileSync(filePath, { encoding: 'base64' });

			console.time('send');
			this.socket.emit('image', imgFile);

			// this.socket.emit('message', Math.sin(this.i * 0.1) * Math.cos(this.i * 0.05 + 4));
			this.i++;
			// console.log(this.i);
			console.timeEnd('send');
		}, 100);
	}

	// Pause emitter
	pause() {
		clearInterval(this.interval);
		this.interval = null;
	}

	// Play emitter
	play() {
		if (!this.interval) {
			this.emit();
		}
	}
}

export default Emitter;
