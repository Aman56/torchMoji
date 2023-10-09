/*playlist choices and setup*/
let playlist1Button = document.getElementById('playlist1Button');
let playlist2Button = document.getElementById('playlist2Button');
let All_song = updateHtml(playlist1);
let song_detail = document.querySelectorAll('.song');
let btn = document.querySelectorAll('.song #play_btn');
let song = document.querySelectorAll('#music');
let activePlaylist = playlist1;

// Add event listeners to the buttons
playlist1Button.addEventListener('click', () => {
  playlist1Button.classList.add('active');
  playlist2Button.classList.remove('active');
  activePlaylist = playlist1;

  All_song = updateHtml(playlist1);
  song_detail = document.querySelectorAll('.song');
  btn = document.querySelectorAll('.song #play_btn');
  song = document.querySelectorAll('#music'); 
});

playlist2Button.addEventListener('click', () => {
  playlist1Button.classList.remove('active');
  playlist2Button.classList.add('active');
  activePlaylist = playlist2;
  All_song = updateHtml(playlist2);
  song_detail = document.querySelectorAll('.song');
  btn = document.querySelectorAll('.song #play_btn');
  song = document.querySelectorAll('#music'); 

});

// Function to update the HTML based on the selected playlist
function updateHtml(selectedPlaylist) {
  let tracks = document.querySelector('.tracks');
  let playlistTracks = selectedPlaylist.tracks;
  tracks.innerHTML = selectedPlaylist.generateHTML();
  return playlistTracks;
}

// Initially, display playlist1 and set playlist1Button as active
updateHtml(playlist1);
playlist1Button.classList.add('active');

//After choosing playlist
let main = document.querySelector('.main');
let topSection = document.querySelector('.top_section');
let searchBar = document.querySelector('.search_bar');
let searchBarInput = document.querySelector('Text');
let searchBarBtn = document.querySelector('.search_bar button');
let track = document.querySelector('.tracks');
let playlistSelector = document.querySelector('.playlist_selector');
let node;
let returnButton;
let generateButton;
let up;
let wavesurfer = WaveSurfer.create({
  container: '#waveform',
  waveColor: '#e3bbee',
  progressColor: '#fe38ab',
  responsive: true,
  hideScrollbar: true,
});

let audio = new Audio('static/music/1.mp3');
wavesurfer.load(audio);
wavesurfer.setMute();

/*popup music player part*/
let p_m_player = document.querySelector('.popup_music_player');
let down_player = document.querySelector('#down_player');
let current_track_name = document.querySelector('#current_track_name');
let current_singer_name = document.querySelector('#current_singer_name');
let rpBtn = document.getElementById('random_play');
let cpBtn = document.getElementById('cycle_play');
let emojiAnimation = document.querySelector('.emoji');
let commentContainer = document.querySelector('.comment-container')
let rotationAngle = 0;

/*controlls part*/
let play_pause_btn = document.querySelector('#play_pause_btn');
let slider = document.querySelector('#slider');
let forward_btn = document.querySelector('#forward_btn');
let backward_btn = document.querySelector('#backward_btn');

/*songs duration*/
let current_duration = document.querySelector('.controlls .progress_part #current_duration');
let total_duration = document.querySelector('.controlls .progress_part #total_duration');

/*small music player part*/
let s_m_player = document.querySelector('.small_music_player');
let playing_img = document.querySelector('.playing_img');
let wave_animation = document.querySelector('.wave_animation');
let up_player = document.querySelector('#up_player');
let song_name = document.querySelector('#song_name');
let artist_name = document.querySelector('#artist_name');

/*default values*/
let is_song_played = false;
let song_status = false;
let index_no = 0;
let randomMode = false;
let cycleMode = false;

/*search values*/
let searchInput = document.getElementById('Text');
let searchButton = document.getElementById('searchButton');
let playlistContainer = document.getElementById('playlistContainer');
let playlist = document.getElementById('playlist');
let searchResults = document.getElementById('searchResults');

//click song and play
// console.log(song_detail);
// // Add event listeners to each div.song element
// song_detail.forEach((songDiv, index) => {
//   songDiv.addEventListener('click', () => {
//     console.log("play")
//     // Pause all other audio elements
//     song.forEach((audio, audioIndex) => {
//       if (audioIndex !== index) {
//         audio.pause();
//       }
//     });

//     // Play the clicked audio element
//     if (song[index].paused) {
//       song[index].play();
//     } else {
//       song[index].pause();
//     }
//   });
// });


// song_detail.forEach((song_detail,index) => {
//   song_detail.addEventListener('click', function(){
//     console.log("clicked")
//     console.log(song_status)
//     s_m_player.style.transform = 'translateY(0px)';
    
//     if (index != index_no) {
//       song_status = false;
//     }
    
//     index_no = index;

//     song[index].currentTime = 0;

//   	if (song_status == false) {
//       const containerContainer = document.getElementById("container-container")
//       removeDiv(containerContainer)
//       play_song();
//   	}
//     else{
//       pause_song();	 
//   	}
//     console.log(song_status)
//   });
// });

// document.querySelectorAll(".tracks .song").forEach(function (song,index) {
//     song.addEventListener("click", function () {
//       console.log("cicked")
//       s_m_player.style.transform = 'translateY(0px)';
//       if (index != index_no) {
//         song_status = false;
//       }
//       console.log(index, song[index])
//       index_no = index;

//       //song[index].currentTime = 0;

//   	  if (song_status == false) {
//         const containerContainer = document.getElementById("container-container")
//         removeDiv(containerContainer)
//         play_song();
//   	  }
//       else{
//         pause_song();	 
//   	  }
//       console.log(song_status)
//     });
//   });

  // Add event listener to the common ancestor (tracks container) of all song elements
  const tracksContainer = document.getElementById("playlist");
  
  tracksContainer.addEventListener("click", function (event) {
    let Thesong = event.target.closest(".song");
    let index;
    if (Thesong) {
      index = Array.from(tracksContainer.querySelectorAll(".song")).indexOf(Thesong);
    }
      s_m_player.style.transform = 'translateY(0px)';
      // console.log("clicked",song)
      // console.log("Thesong",Thesong)
      // console.log(index)
      console.log("song_status",song_status)
      if (index != index_no) {
        song_status = false;
      }
    
      index_no = index;

      song[index].currentTime = 0;

  	  if (song_status == false) {
        const containerContainer = document.getElementById("container-container")
        removeDiv(containerContainer)
        play_song();
  	  }
      else{
        pause_song();	 
  	  }
      console.log("song_status",song_status)
  });




/*pause song*/
function pause_song(){
  wavesurfer.pause();
  song[index_no].pause();
  song_status = false;
  clearInterval(update_second);
  wave_animation.style.opacity = '0';
  play_pause_btn.innerHTML = '<i class="fa fa-play" aria-hidden="true"></i>';
  emojiAnimation.style.animation= 'paused';
}


/*This function will update every 1s*/
 function update_second(){

	  let position = 0;

    // update slider position
		if(!isNaN(song[index_no].duration)){
		   position = song[index_no].currentTime * (100 / song[index_no].duration);
		   slider.value =  position;
	      }

    let durationMinutes = Math.floor(song[index_no].duration / 60);
    let durationSeconds = Math.floor(song[index_no].duration - durationMinutes * 60);
    total_duration.textContent = durationMinutes + ":" + durationSeconds;

    // Calculate the time left and the total duration
    let curr_minutes = Math.floor(song[index_no].currentTime / 60);
    let curr_seconds = Math.floor(song[index_no].currentTime - curr_minutes * 60);
    
    //update the current piece level emoji presentation
    let curr_index = Math.floor(song[index_no].currentTime / 15);
    if (activePlaylist == playlist1){
      updateDiv(curr_index)
    }

    // Add a zero to the single digit time values
    if (curr_seconds < 10) { curr_seconds = "0" + curr_seconds; }
    if (durationSeconds < 10) { durationSeconds = "0" + durationSeconds; }
 
    // Display the updated duration
    current_duration.textContent = curr_minutes + ":" + curr_seconds;

    // function will run when the song is over
	  if (song[index_no].ended) {
      clearInterval(update_second);
  	  wave_animation.style.opacity = '0';
      play_pause_btn.innerHTML = '<i class="fa fa-play" aria-hidden="true"></i>';
      const containerContainer = document.getElementById("container-container")
      removeDiv(containerContainer)
    }
 }
 

/*show popup music player */
up_player.addEventListener('click', function(){
   up = true;
   p_m_player.style.transform = 'translateY(0%)';
   track.style.width = '40%';
   track.style.marginLeft = '7em';
   main.style.alignItems = 'stretch';
   topSection.style.marginLeft = '15em';
   playlistSelector.style.marginRight = '25em';
   searchBar.style.marginLeft = '190px';
   if (searchResults){
    searchResults.style.width = '40%';
    searchResults.style.marginLeft = '7em';
   }
   if (returnButton) {
      returnButton.style.right = '750px';
   }
   if (generateButton){
    generateButton.style.marginLeft = '290px';
   }
});


/* Hide popup music player */
down_player.addEventListener('click', function(){
   up = false
   p_m_player.style.transform = 'translateY(110%)';
   track.style.width = '100%';
   track.style.marginLeft = '0em';
   main.style.alignItems = 'center';
   topSection.style.marginLeft = '0em'
   playlistSelector.style.marginRight = '0em';
   searchBar.style.marginLeft = '380px';
   if (searchResults){
    searchResults.style.width = '100%';
    searchResults.style.marginLeft = '0em';
   }
   if (returnButton) {
    returnButton.style.right = '430px';
   }

   if (generateButton){
    generateButton.style.marginLeft = '900px';
   }
});


/*play pause btn inside the popup Music player*/
play_pause_btn.addEventListener('click', function(){
    if (song_status == false) {
      //commentContainer.classList.remove('paused')
      wavesurfer.playPause();
  	  song[index_no].play();
      song_status = true;
      wave_animation.style.opacity = '1';
  		this.innerHTML = '<i class="fa fa-pause" aria-hidden="true"></i>';
      emojiAnimation.style.animation= 'rotateEmojis 20s linear infinite';
  	}else{
      wavesurfer.pause();
  		song[index_no].pause();
      song_status = false;
      wave_animation.style.opacity = '0';
      this.innerHTML = '<i class="fa fa-play" aria-hidden="true"></i>';
      rotationAngle += getCurrentRotationAngle();
      emojiAnimation.style.animation= 'paused';
      //commentContainer.classList.add('paused')
  	}
});

function getCurrentRotationAngle() {
  const computedStyle = window.getComputedStyle(emojiAnimation);
  const transform = computedStyle.getPropertyValue('transform');
  const matrixValues = transform.match(/matrix.*\((.+)\)/)[1].split(', ');
  const a = matrixValues[0];
  const b = matrixValues[1];
  const currentAngle = Math.round(Math.atan2(b, a) * (180 / Math.PI));
  return currentAngle;
}

// change slider position 
function change_duration(){
	slider_position = song[index_no].duration * (slider.value / 100);
	song[index_no].currentTime = slider_position;
}


/*forward btn (next)*/
forward_btn.addEventListener('click', function(){
  if (randomMode == true){    
		index_no = Math.floor(Math.random() * (All_song.length) )   
	}
  else if (cycleMode == true){
    index_no = index_no;
  }
  else{
    index_no = index_no + 1;
  }

  if (index_no == All_song.length) {
    index_no = 0;
  }
  const containerContainer = document.getElementById("container-container")
  removeDiv(containerContainer)

  song[index_no].currentTime = 0;
  play_song();
});


/*backward btn (previous)*/
backward_btn.addEventListener('click', function(){
  if (randomMode == true){
    index_no = Math.floor(Math.random() * (All_song.length) )        
  }
  else if (cycleMode == true){
    index_no = index_no
  }
  else{
    if (index_no == 0) {
      index_no = All_song.length-1;
    }
    else{
      index_no = index_no -1;
    }
  }
  const containerContainer = document.getElementById("container-container")
  removeDiv(containerContainer)
  song[index_no].currentTime = 0;
  play_song();
  
});


/*play function*/
function play_song(){
  //console.log(song[index_no])
  //console.log(is_song_played)
  wavesurfer.load(song[index_no])
  wavesurfer.playPause();
  song[index_no].play();
  console.log("play_song",All_song,index_no)
  emojiAnimation = updateEmojiContainer(All_song[index_no]);
  if (is_song_played == true) {
    let activeSong = null;  
    song.forEach((audio) => {
      if (audio.classList.contains("active_song")) {
        activeSong = audio;
      }
    });
    console.log(activeSong)
    if (activeSong) {
      activeSong.pause();
      song[index_no].classList.remove("active_song");
    }

  //     console.log(song, song[index_no],document.querySelector(".active_song"))
  //     document.querySelector(".active_song").pause();
  //     document.querySelector(".active_song").classList.remove("active_song");
   }
  else{
        is_song_played = true;
    }
    
  song[index_no].classList.add("active_song");

  song_status = true;
  up = true;
  setInterval(update_second, 1000);
  wave_animation.style.opacity = '1';
  p_m_player.style.transform = 'translateY(0%)';
  track.style.width = '40%';
  track.style.marginLeft = '7em';
  main.style.alignItems = 'stretch';
  topSection.style.marginLeft = '15em';
  playlistSelector.style.marginRight = '25em';
  searchBar.style.marginLeft = '190px';
  if (searchResults){
    searchResults.style.width = '40%';
    searchResults.style.marginLeft = '7em';
  }
  if (returnButton) {
    returnButton.style.right = '750px';
  }
  if (generateButton){
    generateButton.style.marginLeft = '290px';
   }
  song_name.innerHTML = All_song[index_no].name;
  artist_name.innerHTML = All_song[index_no].singer;

  current_track_name.innerHTML = All_song[index_no].name;
  current_singer_name.innerHTML = All_song[index_no].singer;
  play_pause_btn.innerHTML = '<i class="fa fa-pause" aria-hidden="true"></i>';

  const num = Math.floor(song[index_no].duration/15)+1
  if (activePlaylist == playlist1){
    for (let i = 0; i < num; i++) {
      newDiv = CreateDiv(All_song[index_no],num,i)
      placeDiv(newDiv,num,i)
    }
  }
  emojiAnimation.style.animation= 'rotateEmojis 20s linear infinite';
}

/*random play function*/
function RandomPlay (e) {
	randomMode = !randomMode

	var elem = document.getElementById("random_play");

    if (elem.textContent =="random play") {
		elem.textContent = "list loop";
	}
    else {
		elem.textContent = "random play"
	}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
}

/*change play mode*/
rpBtn.addEventListener('click', RandomPlay);

/*cycle play function*/
function CyclePlay (e) {
	cycleMode = !cycleMode
	var elem = document.getElementById("cycle_play");

  if (elem.textContent =="cycle play") {
		elem.textContent = "non-cycle play";
	}
  else {
		elem.textContent = "cycle_play"
	}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
}

/*change play mode*/
cpBtn.addEventListener('click', CyclePlay);

/* add piecewise emoji contaienr */
function CreateDiv(song,num,i){
  if (Math.ceil(i % 4) == 0){
    const newDiv = document.createElement("div");
    //pass the emojis of the song here
    console.log(All_song)
    //const lissy = ["🥰","😉", "🥹","😍", "😗"];
    const lissy = song.emojis
    console.log("emoji div",lissy)
    const node_i = Math.floor(Math.random() * (5))
    lissy_wo_i = lissy.slice(0,node_i).concat(lissy.slice(node_i+1))
    const node_j = Math.floor(Math.random() * (4))

    lissy_wo_j = lissy_wo_i.slice(0,node_j).concat(lissy_wo_i.slice(node_j+1))
    const node_x = Math.floor(Math.random() * (3))
    node = lissy[node_i]+lissy_wo_i[node_j]+lissy_wo_j[node_x]

    const newContent = document.createTextNode(node);
    newDiv.appendChild(newContent);
    return newDiv
  }
  else{
    const newDiv = document.createElement("div");
    const newContent = document.createTextNode(node);
    newDiv.appendChild(newContent);
    return newDiv
  }
  
  /*newDiv.style.left = 80+450/(num-1)*i+'px';*/
  /*console.log(newDiv.style.left)*/
  /*containerContainer.appendChild(newDiv);*/
}

function placeDiv(newDiv,num,i){
  newDiv.classList.add('piece_emoji_container')
  const containerContainer = document.getElementById("container-container")
  newDiv.style.left = 80+450/(num-1)*i+'px';
  newDiv.style.fontSize = "0.3em"
  /*console.log(newDiv.style.left)*/
  containerContainer.appendChild(newDiv);
  newDiv.style.width = '3%';
}

function removeDiv(containerContainer){
  while (containerContainer.firstChild){
    containerContainer.removeChild(containerContainer.firstChild)
    }
}


function updateDiv(currentDivIndex){
  const containerContainer = document.getElementById("container-container")
  const num = Math.floor(song[index_no].duration/15)
  console.log(currentDivIndex,num,containerContainer.children)
  if (currentDivIndex > 0){
    for (let i = 0; i <= num; i++) {
      if (containerContainer.children[i]){
        containerContainer.children[i].style.fontSize = "6px";
      }
    }
  }

  if(containerContainer.children[currentDivIndex]){
    containerContainer.children[currentDivIndex].style.fontSize = "1.2em";
  }
}

searchButton.addEventListener('click', performSearch);

function performSearch() {
  // Clear previous search results
  searchResults.innerHTML = '';

  const query = searchInput.value;
  if (query.trim() === '') {
    alert('Please enter a search emoji or text.');
    return; 
  }
  const results = searchSongs(query);
  console.log(results)
  displaySearchResults(results);
}


function searchSongs(query) {
  const matchingSongs = [];
  let text = query.toLowerCase();
  console.log('query:',text)
  for (const song of All_song) {
    const songEmojis = song.emojis;
    const songTitle = song.name.toLowerCase();
    let isMatch = true;

    for (const term of query) {
      const lowercaseTerm = term.toLowerCase();
      const isEmoji = isEmojiCharacter(term);
      if (isEmoji){
        text = text.replace(term,'')
      }
      if (
        (isEmoji && !songEmojis.includes(term)) ||                   
        (!isEmoji && !songTitle.includes(lowercaseTerm)) ||          
        (!isEmoji && !songTitle.includes(term.toLowerCase()))        
      ) {
        isMatch = false;
        break;
      }
    }

    for (const song of All_song) {
      console.log('after-query:',text)
      if (text){
        if (!songTitle.includes(text)){
          isMatch = false;
          break;
        }
      }
    }

    if (isMatch) {
      matchingSongs.push(song);
    }
  }

  return matchingSongs;
}

// Helper function to check if a given character is an emoji
function isEmojiCharacter(character) {
  const emojiPattern = /^(\p{Emoji_Presentation}|\p{Extended_Pictographic})$/u;
  return emojiPattern.test(character);
}


//generate function
function generateMusic(inputId) {
    const userInput = $("#" + inputId).val();
    console.log("userInput",userInput)
    // Check if user input is empty
    if (userInput.trim() === '') {
        alert('Please enter some input before generating music.');
        return;
    }

    $.ajax({
      type: 'POST',
      url: '/generate_music',
      data: JSON.stringify({ user_input: userInput }),
      contentType: 'application/json',
      success: function(response) {
        if (response.result === 'Music generated successfully') {
          // Display a success message to the user
          //let generatedTrack = response.music
          //console.log("track",generatedTrack)
          alert('Music generated successfully. Please refresh.');
          // let generatedTrack = response.music
          // addSongToPlaylist(generatedTrack);
          // updateHtml();
        }
          console.log(response);
      },
  });
}
function displaySearchResults(results) {
  // Hide the main playlist
  playlistContainer.style.display = 'none';

  let searchResults = document.querySelector('#searchResults');
  for (let i = 0; i < results.length; i++) {
    let Html = ` <div class="song">
        <div class="more">
        <audio src="static/${results[i].path}" id="searchMusic"></audio>
        <div class="song_info">
           <p id="title">${results[i].name}</p>
           <p>${results[i].singer}</p>
        </div>
        <button id="play_btn"><i class="fa fa-angle-right" aria-hidden="true"></i></button>
        </div>
      </div>`;
      searchResults.insertAdjacentHTML("beforeend", Html);
  }
  if (searchResults){
    song_detail = document.querySelectorAll('.song');

    song_detail.forEach((song_detail,index) => {
      song_detail.addEventListener('click', function(){
        if (index != index_no) {
          song_status = false;
        }
        index_no = index;

        if (song_status == false) {
          const containerContainer = document.getElementById("container-container")
          removeDiv(containerContainer)
        }
      });
    });

    //console.log(song_detail)
    const searchSongPlay = searchResults.querySelectorAll('.song');

    searchSongPlay.forEach((searchSongPlay,index) => {
      searchSongPlay.addEventListener('click', function(){
      s_m_player.style.transform = 'translateY(0px)';
      if (index != index_no) {
        song_status = false;
      }
    index_no = results[index].index;
    song[index].currentTime = 0;
      
    if (song_status == false) {
      play_song();
    }
    else{
      pause_song();	 
    }
        });
      });
    }

  //return button
  returnButton = document.createElement('button');
  returnButton.id = 'returnButton';
  returnButton.textContent = 'Return';
  returnButton.addEventListener('click', () => {
    searchResults.innerHTML = '';
    playlistContainer.style.display = 'block'; // Show the main playlist
  });
  searchResults.appendChild(returnButton);
  if (up){
    returnButton.style.right = '750px';
  }

  //generate button
  generateButton = document.createElement('button');
  generateButton.id = 'generateButton';
  generateButton.textContent = "Generate More";

  generateButton.addEventListener('click', () => {
    generateMusic('Text');
  });
  console.log("activePlaylist",activePlaylist)
  if (activePlaylist == playlist2){
    console.log("appended")
    searchResults.appendChild(generateButton);
  }
  if (up){
    generateButton.style.marginLeft = '290px';
  }
}

// Function to update the song level emoji container with the emojis of the currently playing song
function updateEmojiContainer(currentSong) {
  console.log(currentSong)
  const emojiContainer = document.querySelector('.emoji-container');

  if (emojiContainer) {
    // Clear the current content of the emoji container
    emojiContainer.innerHTML = '';

    // Create the emoji structure
    const emojiDiv = document.createElement('div');
    emojiDiv.classList.add('emoji');
    const emojiParagraph = document.createElement('p');

    // Add emojis of the current song to the container
    for (const emoji of currentSong.emojis) {
      const emojiSpan = document.createElement('span');
      emojiSpan.textContent = emoji;
      emojiParagraph.appendChild(emojiSpan);
    }

    // Append the elements to the container
    emojiDiv.appendChild(emojiParagraph);
    emojiContainer.appendChild(emojiDiv);
    return emojiDiv;
  }
}













//not used in current version
$(document).ready(function() {
  function updateUI() {
      //console.log(username)
      if (username) {
          //$("#welcome-container").html(`Welcome, ${username}`);
          //$("#logout-button").show(); // Show the Logout button
      } else {
          $("#welcome-container").html("You haven't logged in");
          $("#logout-button").hide(); // Hide the Logout button
      }
  }

  $.get("/get_username", function(data) {
    username = data.username
    updateUI();
  });

  $("#submit-comment").click(function() {
    const commentText = $("#Input").val();
    if (commentText.trim() === "") return;

    $.post("/submit_comment", { comment: commentText, song_id: index_no}, function(data) {
        username = data.username
        updateUI();
        $("#Input").val("");  // Clear the input field
        // I can choose to update the comment list here or load comments using another AJAX request
        // Example: $.get("/get_comments", { user_id: user_id }, function(data) { ... });
        setTimeout(function() {
          location.reload();
      }, 1500);
    });
});

  $("#logout-button").click(function() {
      $.post("/logout", function(data) {
          username = null
          updateUI()
          //$("#comment-list").empty();  // Clear comment list after logout
          setTimeout(function() {
            location.reload();
        }, 1500);
      });
  });
});

//send song index to flask
function updateSongInfoAndTriggerAjax(index_no) {
  $.ajax({
      type: "GET",
      url: "/",  // Flask route for the index page
      data: { song_id: index_no},
      success: function(response) {
      }
  });
}

setInterval(function() {
  var newIndexNo = index_no;
  updateSongInfoAndTriggerAjax(newIndexNo);
}, 1000);  // Check every 1 second 
