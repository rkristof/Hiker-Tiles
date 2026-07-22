-- Enter/exit Tilemaker
function init_function()
end
function exit_function()
end

node_keys = { "place", "tourism" }

inf_zoom = 99

-- Geofabrik country IDs whose local place names are commonly written in a
-- non-Latin script. Used to decide whether to show "Latin\nOriginal" labels.
local non_latin_countries = {
	["bosnia-herzegovina"] = true,
	["bulgaria"] = true,
	["greece"] = true,
	["kosovo"] = true,
	["macedonia"] = true,
	["montenegro"] = true,
	["serbia"] = true,
	["ukraine"] = true,
}

local country = os.getenv("COUNTRY") or ""
local is_non_latin_country = non_latin_countries[country] or false

function fillWithFallback(value1, value2, value3)
	if value1 ~= "" then
		return value1
	end
	if value2 ~= "" then
		return value2
	end
	return value3
end

-- Build display name: "Latin\nOriginal" for non-Latin-script countries when a
-- Latin alternative is available and differs from the original; otherwise
-- just the best available name.
function buildDisplayName()
	local original = fillWithFallback(Find("name"), Find("name:en"), Find("name:de"))
	if original == "" then
		return ""
	end
	if is_non_latin_country then
		local latin = fillWithFallback(Find("name:en"), Find("int_name"))
		if latin ~= nil and latin ~= "" and latin ~= original then
			return latin .. "\n" .. original
		end
	end
	return original
end

-- Set z_order
function setZOrder()
	local highway = Find("highway")
	local layer = tonumber(Find("layer"))
	local zOrder = 0
	local Z_STEP = 14
	if not (layer == nil) then
		if layer > 7 then
			layer = 7
		elseif layer < -7 then
			layer = -7
		end
		zOrder = zOrder + layer * Z_STEP
	end
	local hwClass = 0
	if highway == "motorway" then
		hwClass = 10
	elseif highway == "trunk" then
		hwClass = 9
	elseif highway == "primary" then
		hwClass = 8
	elseif highway == "secondary" then
		hwClass = 7
	elseif highway == "tertiary" then
		hwClass = 6
	elseif highway == "unclassified" or highway == "residential" or highway == "road" or highway == "motorway_link" or highway == "trunk_link" or highway == "primary_link" or highway == "secondary_link" or highway == "tertiary_link" or highway == "busway" or highway == "bus_guideway" then
		hwClass = 5
	elseif highway == "living_street" or highway == "pedestrian" then
		hwClass = 4
	elseif highway == "service" then
		hwClass = 3
	elseif highway == "footway" or highway == "bridleway" or highway == "cycleway" or highway == "path" or highway == "track" then
		hwClass = 2
	elseif highway == "steps" or highway == "platform" then
		hwClass = 1
	end
	zOrder = zOrder + hwClass
	ZOrder(zOrder)
end

function process_place_layer()
	local place = Find("place")
	local mz = 99
	local kind = place
	local population = Find("population")
	if place == "city" then
		mz = 6
		if population == "" then
			population = "100000"
		end
	elseif place == "town" then
		mz = 7
		if population == "" then
			population = "5000"
		end
	elseif place == "village" then
		mz = 10
		if population == "" then
			population = "100"
		end
	elseif place == "hamlet" then
		mz = 10
		if population == "" then
			population = "50"
		end
	end
	if (place == "city" or place == "town" or place == "village" or place == "hamlet") and Holds("capital") then
		local capital = Find("capital")
		if capital == "yes" then
			mz = 4
			kind = "capital"
		elseif capital == "4" then
			mz = 4
			kind = "state_capital"
		end
	end
	if mz < 99 then
		Layer("place_labels", false)
		MinZoom(mz)
		Attribute("kind", kind)
		Attribute("name", buildDisplayName())
		local populationNum = tonumber(population)
		if populationNum ~= nil then
			ZOrder(populationNum)
		end
	end
end

function node_function()
	-- Layer place_labels
	if Holds("place") and Holds("name") then
		process_place_layer()
	end

	-- Layer pois
	-- Abort here if it was written as POI because Tilemaker cannot write a feature to two layers.
	if process_pois(false) then
		return
	end
end

function zmin_for_area(min_square_pixels, way_area)
	-- Return minimum zoom level where the area of the way/multipolygon is larger than
	-- the provided threshold.
	local circumfence = 40052725.78
	local zmin = (math.log((min_square_pixels * circumfence^2) / (2^16 * way_area))) / (2 * math.log(2))
	return math.floor(zmin)
end

function process_water_polygons(way_area)
	local waterway = Find("waterway")
	local natural = Find("natural")
	local water = Find("water")
	local landuse = Find("landuse")
	local mz = inf_zoom
	local kind = ""
	local is_river = (natural == "water" and water == "river") or waterway == "riverbank"
	if landuse == "reservoir" or landuse == "basin" or (natural == "water" and not is_river) then
		mz = math.max(4, zmin_for_area(0.01, way_area))
		if mz >= 10 then
			mz = math.max(10, zmin_for_area(0.1, way_area))
		end
		if landuse == "reservoir" or landuse == "basin" then
			kind = landuse
		elseif natural == "water" then
			kind = natural
		end
	elseif is_river or waterway == "dock" or waterway == "canal" then
		mz = math.max(4, zmin_for_area(0.1, way_area))
		kind = waterway
		if is_river then
			kind = "river"
		end
	end
	if mz < inf_zoom then
		local way_area = way_area
		Layer("water_polygons", true)
		MinZoom(mz)
		ZOrder(way_area)
	end
end

function process_land()
	local landuse = Find("landuse")
	local natural = Find("natural")
	local wetland = Find("wetland")
	local leisure = Find("leisure")
	local kind = ""
	local mz = inf_zoom
	if landuse == "forest" or landuse == "grass" or landuse == "farmland" then
		kind = landuse
		mz = 5
	elseif natural == "wood" then
		kind = "forest"
		mz = 5
	elseif landuse == "residential" or landuse == "industrial" then
		kind = landuse
		mz = 6
	elseif landuse == "quarry" then
		kind = landuse
		mz = 8
	elseif natural == "glacier" or natural == "bare_rock" then
		kind = natural
		mz = 5
	elseif natural == "sand" or natural == "heath" or natural == "scrub" or natural == "scree" or natural == "shingle" then
		kind = natural
		mz = 7
	elseif natural == "grassland" or landuse == "meadow" then
		kind = "grass"
		mz = 6
	elseif wetland == "swamp" or wetland == "bog" or wetland == "wet_meadow" or wetland == "marsh" then
		kind = wetland
		mz = 7
	elseif leisure == "park" then
		kind = leisure
		mz = 10
	end
	if mz < inf_zoom then
		Layer("land", true)
		MinZoom(mz)
		Attribute("kind", kind)
	end
end

function process_boundary_lines()
	if Holds("type") then
		return
	end
	local min_admin_level = 99
	local disputedBool = false
	while true do
		local rel = NextRelation()
		if not rel and min_admin_level == 99 then
			return
		elseif not rel then
			break
		end
		local admin_level = FindInRelation("admin_level")
		local boundary = FindInRelation("boundary")
		local al = 99
		if admin_level ~= nil and boundary == "administrative" then
			al = tonumber(admin_level)
		end
		if al ~= nil and al >= 2 then
			min_admin_level = math.min(min_admin_level, al)
		end
		if boundary == "disputed" then
			disputedBool = true
		end
	end

	local mz = inf_zoom
	if min_admin_level == 2 then
		mz = 0
	elseif min_admin_level <= 4 then
		mz = 7
	end
	local maritime = Find("maritime")
	local natural = Find("natural")
	local maritimeBool = false
	if maritime == "yes" or natural == "coastline" then
		maritimeBool = true
	end
	if Find("disputed") == "yes" then
		disputedBool = true
	end
	if mz < inf_zoom then
		Layer("boundaries", false)
		MinZoom(mz)
	end
end

function process_streets()
	local min_zoom_layer = 5
	local mz = inf_zoom
	local kind = ""
	local highway = Find("highway")
	local service = Find("service")
	if highway ~= "" then
		if highway == "motorway" or highway == "motorway_link" then
			mz = min_zoom_layer
			kind = "motorway"
		elseif highway == "trunk" or highway == "trunk_link" then
			mz = 6
			kind = "trunk"
		elseif highway == "primary" or highway == "primary_link" then
			mz = 8
			kind = "primary"
		elseif highway == "secondary" or highway == "secondary_link" then
			mz = 9
			kind = "secondary"
		elseif highway == "tertiary" or highway == "tertiary_link" then
			mz = 10
			kind = "tertiary"
		elseif highway == "unclassified" or highway == "residential" or highway == "bus_guideway" or highway == "busway" then
			mz = 12
			kind = highway
		elseif highway == "living_street" or highway == "pedestrian" or highway == "track" then
			mz = 13
			kind = highway
		elseif highway == "service" then
			mz = 14
			kind = highway
		elseif highway == "footway" or highway == "steps" or highway == "path" or highway == "cycleway" then
			mz = 13
			kind = highway
		end
	end
	if mz <= 10 then
		Layer("streets_low", false)
		MinZoom(mz)
		Attribute("kind", kind)
		setZOrder()
	end
	if mz <= 13 then
		Layer("streets_med", false)
		MinZoom(mz)
		Attribute("kind", kind)
		setZOrder()
	end
	if mz < inf_zoom then
		Layer("streets", false)
		MinZoom(mz)
		Attribute("kind", kind)
		local streetName = fillWithFallback(Find("name"), Find("name:en"), Find("name:de"))
		if streetName ~= "" then Attribute("name", streetName) end
		setZOrder()
	end
end

-- Create "pois" layer — viewpoints only.
-- Returns true if the feature is written to that layer.
-- Returns false if it was no POI we are interested in.
function process_pois(polygon)
	if Find("tourism") ~= "viewpoint" then
		return false
	end
	if polygon then
		LayerAsCentroid("pois")
	else
		Layer("pois", false)
	end
	MinZoom(14)
	Attribute("tourism", "viewpoint")
	Attribute("name", fillWithFallback(Find("name"), Find("name:en"), Find("name:de")))
	return true
end

function way_function()
	local area = Area()
	local area_tag = Find("area")
	local type_tag = Find("type")
	-- Way/Relation is explicitly tagged as area.
	local area_yes_multi_boundary = (area_tag == "yes" or type_tag == "multipolygon" or type_tag == "boundary")
	-- Boolean flags for closed ways in cases where features can be mapped as line or area
	-- If closed ways are assumed to be polygons by default except tagged with area=no
	local is_area = (area_yes_multi_boundary or (area > 0 and area_tag ~= "no"))
	-- If closed ways are assumed to be rings by default except tagged with area=yes, type=multipolygon or type=boundary
	local is_area_default_linear = area_yes_multi_boundary

	-- Layers water_polygons
	if is_area and (Holds("waterway") or Holds("natural") or Holds("landuse")) then
		process_water_polygons(area)
	end

	-- Layer land
	if is_area and (Holds("landuse") or Holds("natural") or Holds("wetland") or Holds("leisure")) then
		process_land()
	end

	-- Layer boundaries
	process_boundary_lines()

	-- Layer streets
	if not is_area_default_linear and Holds("highway") then
		process_streets()
	end

	-- Layer pois
	local is_poi = false
	if is_area then
		is_poi = process_pois(true)
	end

	-- Abort here if it was written as POI because Tilemaker cannot write a feature to two layers.
	if is_poi then
		return
	end
end

-- Check that admin_level is 2, 3 or 4
function admin_level_valid(admin_level, is_unset_valid)
	return (is_unset_valid and admin_level == "") or admin_level == "2" or admin_level == "3" or admin_level == "4"
end

---- Accept boundary relations
function relation_scan_function()
	if Find("type") ~= "boundary" then
		return
	end
	local boundary = Find("boundary")
	if boundary == "administrative" then
		if admin_level_valid(Find("admin_level"), false) then
			Accept()
		end
	elseif boundary == "disputed" then
		if admin_level_valid(Find("admin_level"), true) then
			Accept()
		end
	end
end