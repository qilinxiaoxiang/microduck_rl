// Timed UI inputs use simulated time, never a guessed wall-clock duration.
export function timedForward({phase,paused,time},seconds){
  if(!Number.isFinite(time)||time<0||![.5,2].includes(seconds))throw Error('Invalid timed walking input');
  if(phase!=='manual'||paused)return null;
  return {time,type:seconds===.5?'forward-half-second':'forward-two-seconds',until:time+seconds};
}
