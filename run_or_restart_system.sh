#!/bin/bash

function log {
  tput setaf 2
  echo "$1"
  tput sgr0
}

# By default, we will do up.
# if second arg exists, we will do restart.
# If second arg equals stop, we will stop.

if [ -z $1 ] 
then
    COMMAND="up"
    # Run the timesync script
else
    COMMAND="restart"
fi

if [ $1 = "stop" ]
then
    COMMAND="stop"
fi

log "Command is:"
log $COMMAND


#function get_modules {
# curl -s http://localhost:3000/api/get_active_leases | jq '.[] | .host' | grep modul | xargs -L1
#}
#
#modules=get_modules()
#for module in $modules; do
# echo $module
#done

xhost local:root

#log "Restart DHCP"
#docker restart glass-isc-dhcp


docker-compose -f /home/boris/Desktop/AllNet/docker-compose.yml $COMMAND &
sleep 3

#############################################
if [ ! $COMMAND = "stop" ]
then
    konsole --noclose -e "docker exec -it deepspeech-container /bin/run-jupyter"
fi

