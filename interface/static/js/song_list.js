class Playlist {
  constructor() {
    this.tracks = [];
  }

  addTrack(track) {
    this.tracks.push(track);
  }

  clearTracks() {
    this.tracks = [];
  }

  generateEmojiHTML() {
    let emojiHTML = '';
    for (const track of this.tracks) {
      for (const emoji of track.emojis) {
        emojiHTML += `<span>${emoji}</span>`;
      }
    }
    return emojiHTML;
  }

  generateHTML() {
    let html = '';
    for (let i = 0; i < this.tracks.length; i++) {
      const track = this.tracks[i];
      let audioSrc = `static/${track.path}`;
      html += `
        <div class="song">
          <div class="more">
            <audio src="${audioSrc}" id="music" type="audio/mpeg"></audio>
            <div class="song_info">
              <p id="title">${track.name}</p>
              <p>${track.singer}</p>
            </div>
            <button id="play_btn"><i class="fa fa-angle-right" aria-hidden="true"></i></button>
          </div>
        </div>
      `;
    }
    return html;
  }
}

// Create instances of the Playlist class for each playlist
let playlist1 = new Playlist();
let playlist2 = new Playlist();

// Add tracks to the playlists
playlist1.addTrack({
  index: 0,
  name: "Embrace",
  path: "music/1.mp3",
  singer: "ItsWatR",
  emojis: ["😎", "😉", "🥹", "😍","😀"]
});

playlist1.addTrack({
  index: 1,
  name: "Happy Day",
  path: "music/2.mp3",
  singer: "Stockaudios",
  emojis: ["😎", "😉", "🥹", "😍", "😀"]
});

playlist1.addTrack({
  index: 2,
  name: "Fun Life",
  path: "music/3.mp3",
  singer: "FASSounds",
  emojis: ["😉", "🥹", "😍", "🥰", "🤩"]
});

playlist1.addTrack({
  index: 3,
  name: "First Steps",
  path: "music/4.mp3",
  singer: "SoulProdMusic",
  emojis: ["😉", "🥹", "😎", "🤩", "🥰"]
});

playlist1.addTrack({
  index: 4,
  name: "A Small Miracle",
  path: "music/5.mp3",
  singer: "Romarecord1973",
  emojis: ["🥰", "🥹", "😍", "😭", "🤩"]
});

playlist1.addTrack({
  index: 5,
  name: "Coniferous forest",
  path: "music/6.mp3",
  singer: "orangery",
  emojis: ["🥰", "🥹", "😍", "😭", "😀"]
});

playlist1.addTrack({
  index: 6,
  name: "Risk",
  path: "music/7.mp3",
  singer: "StudioKolomna",
  emojis: ["🥰", "🥹", "💕", "😭","😫"]
});

playlist1.addTrack({
  index: 7,
  name: "Smoke",
  path: "music/8.mp3",
  singer: "SoulProdMusic",
  emojis: ["💕", "🥹", "😍", "😭","😜"]
});

playlist1.addTrack({
  index: 8,
  name: "Waterfall",
  path: "music/9.mp3",
  singer: "RomanSenyMusic",
  emojis: ["💕", "🥹", "😍", "😭","😜"]
});

// Function to fetch and add pregenerated music to playlist2
async function addPregeneratedMusicToPlaylist() {
  try {
    let response = await fetch('/static/pregenerated_music/');
    let filenames = await response.json(); 
    filenames = filenames.filter((filename) => filename.endsWith('.mp3') || filename.endsWith('.wav'));
    const emojiPattern = /[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F700}-\u{1F77F}\u{1F780}-\u{1F7FF}\u{1F800}-\u{1F8FF}\u{1F900}-\u{1F9FF}\u{1FA00}-\u{1FA6F}\u{1FA70}-\u{1FAFF}\u{1FB00}-\u{1FBFF}\u{1FC00}-\u{1FCFF}\u{1FD00}-\u{1FDFF}\u{1FE00}-\u{1FEFF}\u{1FF00}-\u{1FFFF}]/gu;
    filenames.forEach((filename, index) => {
      const name = filename.split('.')[0];
      const emojiMatches = [...filename.matchAll(emojiPattern)];
      const emojis = emojiMatches.map(match => match[0]);
      const singer = 'MusicGen';
      playlist2.addTrack({
        index: index,
        name: name,
        path: `pregenerated_music/${filename}`, 
        singer: singer,
        emojis: emojis,
      });
    });

  } catch (error) {
    console.error('Error fetching pregenerated music:', error);
  }
}

// Call the function to add pregenerated music to playlist2
addPregeneratedMusicToPlaylist();



// Update the HTML for the selected playlist
// function updateHtml(selectedPlaylist) {
//   const tracks = document.querySelector('.tracks');
//   tracks.innerHTML = selectedPlaylist.generateHTML();
// }

window.playlist1 = playlist1;
window.playlist2 = playlist2;