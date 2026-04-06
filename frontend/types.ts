export interface CameraDevice {
  deviceId: string;
  label: string;
}

export enum ConnectionStatus {
  DISCONNECTED = 'DISCONNECTED',
  CONNECTED = 'CONNECTED',
  CAPTURING = 'CAPTURING',
  UPLOADING = 'UPLOADING',
  ERROR = 'ERROR'
}

export interface ServerCommand {
  action: 'idle' | 'capture';
  id?: string; // unique ID to prevent double capture
}