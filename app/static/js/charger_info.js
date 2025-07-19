function fetchStatuses() {
    const url = WIDGET_MODE ? "/dashboard/widget/statuses" : "/dashboard/statuses";
    fetch(url)
        .then(res => res.json())
        .then(data => {
            for (const [chargerId, info] of Object.entries(data)) {
                const statusEl = document.getElementById(`status-${chargerId}`);
                const energyEl = document.getElementById(`energy-${chargerId}`);
                const costEl = document.getElementById(`cost-${chargerId}`);
                const secondsEl = document.getElementById(`seconds-${chargerId}`);

                if (statusEl) statusEl.textContent = info.status;
                if (energyEl) energyEl.textContent = info.energy.toFixed(2);
                if (costEl) costEl.textContent = info.cost.toFixed(2);
                if (secondsEl) secondsEl.textContent = formatDuration(info.seconds);

                // Get the card container
                const card = document.querySelector(`[data-id='${chargerId}']`)?.parentElement?.parentElement;
                if (card) {
                    if (info.status === "Charging") {
                        card.classList.add("glow");
                    } else {
                        card.classList.remove("glow");
                    }
                }
            }
        })
        .catch(err => console.error("Failed to fetch statuses:", err));
}

setInterval(fetchStatuses, 60000);  // every 5 seconds
fetchStatuses();  // run immediately on load


// Charger map
const mapInstances = {};
const mapMarkers = {};

function toggleMap(el) {
    const chargerId = el.dataset.id;
    const lat = parseFloat(el.dataset.lat);
    const lng = parseFloat(el.dataset.lng);
    const mapContainer = document.getElementById(`map-${chargerId}`);

    if (mapContainer.classList.contains('hidden')) {
        mapContainer.classList.remove('hidden');

        if (!mapInstances[chargerId]) {
            const map = L.map(`map-${chargerId}`, {
                fullscreenControl: true,
            }).setView([lat, lng], 13);

            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; OpenStreetMap contributors',
            }).addTo(map);

            const marker = L.marker([lat, lng]).addTo(map);
            mapInstances[chargerId] = map;
            mapMarkers[chargerId] = marker;

            // Custom Center Button
            const centerControl = L.control({ position: 'topright' });
            centerControl.onAdd = function () {
                const btn = L.DomUtil.create('button', 'leaflet-bar leaflet-control leaflet-control-custom');
                btn.innerHTML = '🎯';
                btn.title = 'Center on Marker';
                btn.style.backgroundColor = 'white';
                btn.style.width = '34px';
                btn.style.height = '34px';
                btn.style.cursor = 'pointer';

                btn.onclick = function (e) {
                    e.preventDefault();
                    e.stopPropagation();
                    map.setView([lat, lng], 13);
                };

                return btn;
            };

            centerControl.addTo(map);
        } else {
            setTimeout(() => mapInstances[chargerId].invalidateSize(), 100);
        }
    } else {
        mapContainer.classList.add('hidden');
    }
}
