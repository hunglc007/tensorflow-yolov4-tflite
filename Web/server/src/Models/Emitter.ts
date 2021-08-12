import { Socket } from 'socket.io';

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
			this.socket.emit('message', Math.sin(this.i * 0.1) * Math.cos(this.i * 0.05 + 4));
			this.i++;
			console.log(this.i);
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
