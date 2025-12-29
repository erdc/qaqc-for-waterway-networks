import sys
import time
import traceback
import networkx as nx
from collections import Counter
from datetime import (date)
from difflib import get_close_matches
from functools import wraps
from qgis import processing
from qgis.PyQt.QtCore import (
    QCoreApplication,
    QVariant,
)
from qgis.analysis import QgsGeometrySnapper
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsDistanceArea,
    QgsFeature,
    QgsFeatureRequest,
    QgsFeatureSink,
    QgsField,
    QgsGeometry,
    QgsMapLayer,
    QgsPointXY,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputString,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterDistance,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProject,
    QgsSpatialIndex,
    QgsUnitTypes,
    QgsVectorLayer,
    QgsVectorLayerUtils,
    QgsWkbTypes,
    edit,
    Qgis,
    QgsRectangle,
    QgsExpression,
    QgsPoint,
    QgsLineString
)

CONTEXT = None
FEEDBACK = None
START_TIME = None
DISTANCE_AREA = None
STEP = 0
TOTAL_STEPS = 34


class QgsProcessingCanceledException(Exception):
    pass


def check_feedback():
    if FEEDBACK.isCanceled():
        raise QgsProcessingCanceledException("")


def cancel_on_entry(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if FEEDBACK.isCanceled():
            raise QgsProcessingCanceledException("")
        return func(*args, **kwargs)
    return wrapper


def interrupt_on_cancel(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        def tracer(frame, event, arg):
            if event == 'line' and FEEDBACK.isCanceled():
                raise QgsProcessingCanceledException("")
            return tracer
        sys.settrace(tracer)
        try:
            return func(*args, **kwargs)
        finally:
            sys.settrace(None)
    return wrapper


class DisconnectedIslands(object):
    def __init__(self, l, t):
        self.layer = l
        self.disconnected_islands_tolerance = t

    def run(self, tolerance):
        attr_idx = self.layer.fields().indexFromName("networkGrp")
        if attr_idx == -1:
            self.layer.startEditing()
            self.layer.dataProvider().addAttributes([QgsField("networkGrp", QVariant.Int)])
            self.layer.commitChanges()
            attr_idx = self.layer.fields().indexFromName("networkGrp")
        G = nx.Graph()
        if tolerance == 0:
            tolerance = self.disconnected_islands_tolerance
        self.layer.startEditing()
        for feat in self.layer.getFeatures():
            check_feedback()

            self.layer.changeAttributeValue(feat.id(), attr_idx, -1)
            geom = feat.geometry()
            QgsGeometry.convertToSingleType(geom)
            if not geom.isNull():
                line = geom.asPolyline()
                for i in range(len(line) - 1):
                    check_feedback()

                    G.add_edges_from([((int(line[i][0] / tolerance), int(line[i][1] / tolerance)),
                                       (int(line[i + 1][0] / tolerance), int(line[i + 1][1] / tolerance)),
                                       {'fid': feat.id()})])
        self.layer.commitChanges()
        connected_components = list(G.subgraph(c) for c in nx.connected_components(G))
        fid_comp = {}
        for i, graph in enumerate(connected_components):
            check_feedback()

            for edge in graph.edges(data=True):
                check_feedback()

                fid_comp[edge[2].get('fid', None)] = i
        count_map = {}
        for v in fid_comp.values():
            check_feedback()

            count_map[v] = count_map.get(v, 0) + 1
        isolated = [k for k, v in fid_comp.items() if count_map[v] == 1]
        self.layer.selectByIds(isolated)
        self.layer.startEditing()
        for (fid, group) in fid_comp.items():
            check_feedback()

            self.layer.changeAttributeValue(fid, attr_idx, group)
        self.layer.commitChanges()
        return self.layer, [i for i in set(fid_comp.values()) if i > 0]

@cancel_on_entry
def run_alg(alg, params, delete_input=True, is_child_alg=True):
    result = processing.run(
        alg, {**params, 'OUTPUT': 'TEMPORARY_OUTPUT'},
        is_child_algorithm=is_child_alg, context=CONTEXT
    )
    output = CONTEXT.getMapLayer(result['OUTPUT'])
    if delete_input and params.get('INPUT'):
        delete_layer(params["INPUT"])
    return output


@cancel_on_entry
def flatten_collection(items):
    for item in items:
        check_feedback()
        if isinstance(item, (list, tuple)):
            yield from flatten_collection(item)
        else:
            yield item


@interrupt_on_cancel
def format_time():
    sec = int(time.time() - START_TIME)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"(Time: {h:02}:{m:02}:{s:02})"


@interrupt_on_cancel
def iterate_step(amount=0):
    global STEP
    if amount > 0:
        STEP += amount
    else:
        STEP += 1


@interrupt_on_cancel
def update_progress(changelog=""):
    current_time = format_time()
    iterate_step()
    percent = int((STEP / TOTAL_STEPS) * 100)
    FEEDBACK.setProgress(percent)
    if changelog:
        FEEDBACK.setProgressText(f"{changelog} {current_time}")


@interrupt_on_cancel
def add_unique_field(layer, name, type_):
    if name not in [field.name() for field in layer.fields()]:
        layer.dataProvider().addAttributes([QgsField(name, type_)])
        layer.updateFields()


@interrupt_on_cancel
def copy_layer(layer, expr=""):
    if expr != "":
        return layer.materialize(QgsFeatureRequest().setFilterExpression(expr))
    return layer.materialize(QgsFeatureRequest())


@interrupt_on_cancel
def delete_layer(layer):
    if isinstance(layer, QgsMapLayer):
        QgsProject.instance().removeMapLayer(layer.id())
    del layer


@cancel_on_entry
def remove_features_by_expression(layer, expression):
    with edit(layer):
        for feature in layer.getFeatures(
                QgsFeatureRequest().setFilterExpression(expression)
        ):
            check_feedback()
            layer.deleteFeature(feature.id())


@cancel_on_entry
def remove_feature_attribute_by_name(layer, attr_name):
    index_from_name = layer.fields().indexFromName(attr_name)
    if index_from_name != -1:
        layer.dataProvider().deleteAttributes([index_from_name])
        layer.updateFields()


@cancel_on_entry
def clean_layer(layer):
    fix_geometries = ('native:fixgeometries', {'INPUT': {}, 'METHOD': 1})
    remove_null_geometries = (
        'native:removenullgeometries', {'INPUT': {}, 'REMOVE_EMPTY': False}
    )
    remove_duplicate_vertices = (
        'native:removeduplicatevertices',
        {'INPUT': {}, 'TOLERANCE': 1e-6, 'USE_Z_VALUE': False}
    )
    delete_duplicate_geometries = (
        'native:deleteduplicategeometries', {'INPUT': {}}
    )
    clean_steps = [
        fix_geometries, remove_null_geometries,
        fix_geometries, remove_duplicate_vertices,
        fix_geometries, delete_duplicate_geometries,
        fix_geometries
    ]
    for alg, params in clean_steps:
        check_feedback()

        params["INPUT"] = layer
        layer = run_alg(alg, params)
    return layer


@cancel_on_entry
def split_lines_by_points(
        line_layer,
        point_layer,
        min_gap=0.0,           # map units; 0 disables thinning
        end_epsilon=1e-9):     # map units; guards against endpoint splits

    crs = line_layer.sourceCrs()
    fields = line_layer.fields()
    out = QgsVectorLayer(f"LineString?crs={crs.authid()}", "split_lines", "memory")
    pr = out.dataProvider(); pr.addAttributes(fields); out.updateFields()

    def uniq_positions(geom, pts):
        # get monotonically increasing positions along the line, filtered
        pos = []
        for p in pts:
            check_feedback()

            q = geom.nearestPoint(QgsGeometry.fromPointXY(p))
            d = geom.lineLocatePoint(q)
            pos.append((d, QgsPointXY(q.asPoint())))
        pos.sort(key=lambda t: t[0])

        # drop points too close to ends
        l = geom.length()
        filtered = []
        last_d = None
        for d, pt in pos:
            check_feedback()

            if d is None or d <= end_epsilon or (l - d) <= end_epsilon:
                continue
            if last_d is None or (min_gap == 0.0) or abs(d - last_d) >= min_gap:
                filtered.append((d, pt))
                last_d = d
        return [pt for _, pt in filtered]

    # index points by bbox to reduce scans
    pidx = QgsSpatialIndex(point_layer.getFeatures(),
                           flags=QgsSpatialIndex.FlagStoreFeatureGeometries)

    req = QgsFeatureRequest()
    new_feats = []
    for lf in line_layer.getFeatures():
        check_feedback()

        g = lf.geometry()
        if not g or g.isEmpty():
            continue

        # fetch all candidate points by bbox, no artificial buffers
        cands = []
        for pid in pidx.intersects(g.boundingBox()):
            check_feedback()

            pf = point_layer.getFeature(pid)
            pg = pf.geometry()
            if not pg or pg.isEmpty():
                continue
            # accept only points actually on/near the line via nearestPoint projection
            p = pg.asPoint()
            proj = g.nearestPoint(QgsGeometry.fromPointXY(p))
            if proj is None or proj.isEmpty():
                continue
            cands.append(QgsPointXY(proj.asPoint()))

        if not cands:
            nf = QgsFeature(fields); nf.setAttributes(lf.attributes()); nf.setGeometry(g)
            new_feats.append(nf); continue

        split_pts = uniq_positions(g, cands)

        parts = [g]
        if split_pts:
            # split sequentially; QGIS splitGeometry tolerates tiny duplicates
            for pt in split_pts:
                check_feedback()

                nxt = []
                for seg in parts:
                    check_feedback()

                    res, geoms, _ = seg.splitGeometry([pt], False)
                    if res == 0:
                        nxt.append(seg)
                        if geoms: nxt.extend(geoms)
                    else:
                        nxt.append(seg)
                parts = nxt

        for piece in parts:
            check_feedback()

            nf = QgsFeature(fields); nf.setAttributes(lf.attributes()); nf.setGeometry(piece)
            new_feats.append(nf)

    pr.addFeatures(new_feats); out.updateExtents()
    return out


@cancel_on_entry
def get_internal_connected_features(layer):
    # build spatial index
    spatial_index = QgsSpatialIndex(
        layer.getFeatures(),
        flags=QgsSpatialIndex.FlagStoreFeatureGeometries
    )
    internal_feats = []
    for feature in layer.getFeatures():
        check_feedback()
        geom = feature.geometry()
        for neighbor_id in spatial_index.intersects(geom.boundingBox()):
            check_feedback()
            if feature.id() >= neighbor_id:
                continue
            neighbor = layer.getFeature(neighbor_id)
            intersection = geom.intersection(neighbor.geometry())
            if intersection and not intersection.isEmpty():
                geom_vertices = list(geom.vertices())
                if intersection.type() == QgsWkbTypes.PointGeometry:
                    for pt in intersection.asMultiPoint() \
                            if intersection.isMultipart() \
                            else [intersection.asPoint()]:
                        check_feedback()
                        pt_xy = QgsPointXY(pt)
                        if not (
                                pt_xy == QgsPointXY(geom_vertices[0])
                                or
                                pt_xy == QgsPointXY(geom_vertices[-1])
                        ):
                            internal_feats.append(neighbor)
    return internal_feats


@cancel_on_entry
def split_input_with_ncf_lines(bypass_split, layer, ncf_layer):
    if not bypass_split:
        # convert ncf layer from polygons to lines
        ncf_lines_layer = run_alg(
            "native:polygonstolines",
            {"INPUT": ncf_layer},
            False
        )
        update_progress("- Converted NCF polygons to lines")
        # get intersection points between waterway lines and ncf lines
        pts = run_alg(
            "native:lineintersections", {
                "INPUT": layer,
                "INTERSECT": ncf_lines_layer,
                "INPUT_FIELDS": [],
                "INTERSECT_FIELDS": [],
                "INPUT_FIELDS_PREFIX": "",
                "INTERSECT_FIELDS_PREFIX": ""
            }
        )
        update_progress("- Found intersection points between waterway lines and NCF polygons")
        # split lines by points
        layer = split_lines_by_points(
            layer, pts
        )
        update_progress("- Split waterway lines at intersection points")
        delete_layer(ncf_lines_layer)
    else:
        for count in range(3):
            check_feedback()

            update_progress()
    return layer


@cancel_on_entry
def recalculate_feature_lengths(layer):
    with edit(layer):
        for feature in layer.getFeatures():
            check_feedback()
            feature["LenMiles"] = round(
                DISTANCE_AREA.convertLengthMeasurement(
                    feature.geometry().length(), QgsUnitTypes.DistanceMiles
                ),
                4
            )
            layer.updateFeature(feature)
    update_progress("- Recalculated feature lengths")


@cancel_on_entry
def fix_link_types(layer, flagged):
    link_types = [
        "CPT", "Centerline", "Coastal-connect", "Inland",
        "Great Lakes/St", "International", "Internat River"
    ]
    joined_link_types = "'" + "', '".join(link_types) + "'"
    link_type_expr = f"(LinkType NOT IN ({joined_link_types}) OR LinkType IS NULL) AND NOT Name ILIKE '%Manual Connection%'"
    with edit(layer):
        for feature in layer.getFeatures(
                QgsFeatureRequest().setFilterExpression(link_type_expr)
        ):
            check_feedback()
            if feature["LinkType"] is None:
                flagged.append((feature["Name"], "Null LinkType"))
                continue
            link_type = str(feature["LinkType"])
            if 'lock' not in link_type.lower():
                matches = get_close_matches(
                    link_type, link_types, n=1, cutoff=0.5
                )
                if matches:
                    feature["LinkType"] = matches[0]
                    layer.updateFeature(feature)
                else:
                    flagged.append((feature["Name"], "Invalid LinkType"))
    update_progress("- Fixed LinkType typos")


@cancel_on_entry
def get_geom_len_mi(geom):
    return round(
        DISTANCE_AREA.convertLengthMeasurement(
            geom.length(), QgsUnitTypes.DistanceMiles
        ), 4
    )


@cancel_on_entry
def remove_empty_and_short_geometries(layer, min_geom_length, protected_ids=None):
    protected_ids = protected_ids or set()
    with edit(layer):
        for feature in layer.getFeatures():
            check_feedback()

            if feature.id() in protected_ids:
                continue
            g = feature.geometry()
            if g.isNull() or g.isEmpty() or not feature.hasGeometry() or get_geom_len_mi(g) <= min_geom_length:
                if not str(feature["Name"]).startswith('Manual Connection'):
                    layer.deleteFeature(feature.id())

    update_progress("- Removed empty & very short geometries")


@cancel_on_entry
def snap_geometries_with_snapper(layer, tolerance_deg):
    # make a static reference copy so snapping doesn't cascade while we edit
    ref_layer = copy_layer(layer)
    snapper = QgsGeometrySnapper(ref_layer)

    # cache original geometries in case we want to guard against degenerates
    feats = list(layer.getFeatures())

    # EndPointPreferClosest == 5; only endpoints move, using closest-point mode
    snapped_feats = snapper.snapFeatures(
        feats,
        tolerance_deg,
        QgsGeometrySnapper.EndPointPreferClosest
    )

    # apply snapped geometries back onto the original layer
    with edit(layer):
        for f in snapped_feats:
            check_feedback()

            fid = f.id()
            new_geom = f.geometry()
            if not new_geom or new_geom.isEmpty():
                # keep original if snapper somehow produced an empty geometry
                continue

            # guard against self-loops / collapsed lines in deg units
            # (very small length -> keep original geometry instead)
            if new_geom.length() < 1e-10:
                continue

            layer.changeGeometry(fid, new_geom)

    delete_layer(ref_layer)

    update_progress("- Snapped geometries")
    return layer


@cancel_on_entry
def snap_geometries(layer, tolerance):
    layer = run_alg(
        "native:snapgeometries",
        {
            'INPUT': layer, "REFERENCE_LAYER": layer,
            "TOLERANCE": tolerance, "BEHAVIOR": 6
        })
    update_progress("- Snapped geometries")
    return layer


@cancel_on_entry
def reconnect_islands(layer, island_tol, snapping_tolerance):
    default_island_tol = 0.000001
    disconnected_islands = DisconnectedIslands(layer, default_island_tol)
    disconnected_layer, islands = disconnected_islands.run(island_tol)

    tolerance = snapping_tolerance
    skipped_fids = set()

    # helpers
    def _near_dateline_deg(lon, deg_tol):
        return abs(abs(lon) - 180.0) <= deg_tol

    def _wrap_lon_diff_deg(lon1, lon2):
        # minimal |Δλ| on a circle (degrees)
        d = abs((lon1 - lon2 + 540.0) % 360.0 - 180.0)
        return d

    def _virtually_connected_across_dateline(isle_feat, seam_pts_main, deg_tol):
        g = isle_feat.geometry()
        if not g or g.isEmpty():
            return False
        for v1 in g.vertices():
            check_feedback()

            lon1, lat1 = float(v1.x()), float(v1.y())
            if not _near_dateline_deg(lon1, deg_tol):
                continue
            for p2 in seam_pts_main:
                check_feedback()

                lon2, lat2 = p2.x(), p2.y()
                if abs(lat1 - lat2) > deg_tol:
                    continue
                if _wrap_lon_diff_deg(lon1, lon2) <= deg_tol:
                    return True
        return False

    while islands:
        check_feedback()

        mainland_feats = []
        islands_set = set(islands)
        island_groups = {gid: [] for gid in islands_set}
        for f in disconnected_layer.getFeatures():
            check_feedback()

            grp = f["networkGrp"]
            if grp == 0:
                mainland_feats.append(f)
            elif grp in islands_set and f.id() not in skipped_fids:
                island_groups[grp].append(f)

        active_islands = [gid for gid, feats in island_groups.items() if feats]
        if not active_islands:
            break

        mainland_idx = QgsSpatialIndex(flags=QgsSpatialIndex.FlagStoreFeatureGeometries)
        mainland_idx.addFeatures(mainland_feats)

        mainland_vertices = {}
        seam_pts_mainland = []
        for mf in mainland_feats:
            check_feedback()

            mg = mf.geometry()
            if not mg or mg.isEmpty():
                continue

            mainland_vertices.setdefault(mf.id(), [])

            for v in mg.vertices():
                check_feedback()

                mainland_vertices[mf.id()].append(QgsPointXY(v))

                x = float(v.x())
                if _near_dateline_deg(x, tolerance):
                    seam_pts_mainland.append(QgsPointXY(x, float(v.y())))

        with edit(disconnected_layer):
            check_feedback()

            new_features = []
            island_vertices_cache = {}

            for island in active_islands:
                check_feedback()

                island_group = island_groups.get(island, [])
                if bool(seam_pts_mainland) and any(_virtually_connected_across_dateline(f, seam_pts_mainland, tolerance) for f in island_group):
                    skipped_fids.update(f.id() for f in island_group)
                    continue  # do not create a connector, treat as connected on a globe

                min_dist = float("inf")
                closest_island_feat = None
                closest_island_p = None
                closest_main_p = None

                for island_feat in island_group:
                    check_feedback()

                    nn_ids = mainland_idx.nearestNeighbor(island_feat.geometry(), 1, tolerance)
                    if not nn_ids:
                        continue

                    mainland_fid = nn_ids[0]

                    if island_feat.id() not in island_vertices_cache:
                        island_vertices_cache[island_feat.id()] = [
                            QgsPointXY(v) for v in island_feat.geometry().vertices()
                        ]

                    island_points = island_vertices_cache.get(island_feat.id())
                    main_points = mainland_vertices.get(mainland_fid)
                    if not island_points or not main_points:
                        continue
                    if len(island_points) <= 1 or len(main_points) <= 1:
                        continue

                    for p1 in (island_points[0], island_points[-1]):
                        check_feedback()

                        for p2 in (main_points[0], main_points[-1]):
                            check_feedback()

                            d = DISTANCE_AREA.measureLine(p1, p2)
                            if d < min_dist:
                                min_dist = d
                                closest_island_feat = island_feat
                                closest_island_p = p1
                                closest_main_p = p2

                if min_dist != float("inf") and closest_island_feat is not None:
                    connector = QgsFeature(disconnected_layer.fields())
                    connector.setGeometry(
                        QgsGeometry.fromPolylineXY([closest_island_p, closest_main_p])
                    )
                    connector.setAttributes(closest_island_feat.attributes())
                    new_features.append(connector)

            if new_features:
                disconnected_layer.addFeatures(new_features)

        disconnected_layer, islands = disconnected_islands.run(island_tol)
        tolerance += snapping_tolerance

    remove_features_by_expression(disconnected_layer, "networkGrp = -1")
    update_progress("- Reconnected disconnected islands (antimeridian-aware)")
    return disconnected_layer


@cancel_on_entry
def fix_overlaps(layer):
    spatial_index = QgsSpatialIndex(
        layer.getFeatures(),
        flags=QgsSpatialIndex.FlagStoreFeatureGeometries
    )
    checked = set()
    with edit(layer):
        for feature in layer.getFeatures():
            check_feedback()
            fid = feature.id()
            feature_geom = feature.geometry()
            neighbor_ids = spatial_index.intersects(feature_geom.boundingBox())
            for neighbor_id in neighbor_ids:
                check_feedback()
                # skip already checked feature combos
                if fid >= neighbor_id or (fid, neighbor_id) in checked:
                    continue
                checked.add((fid, neighbor_id))
                neighbor_geom = layer.getFeature(neighbor_id).geometry()
                # create the geometry engine and prepare geometry
                geom_engine = QgsGeometry.createGeometryEngine(
                    feature_geom.constGet()
                )
                geom_engine.prepareGeometry()
                # check if there is a partial overlap
                if geom_engine.overlaps(neighbor_geom.constGet()):
                    inter = feature_geom.intersection(neighbor_geom)
                    if inter and not inter.isEmpty():
                        trimmed = feature_geom.difference(inter)
                        if trimmed and not trimmed.isEmpty():
                            layer.changeGeometry(fid, trimmed)
                # check if feature contains neighbor
                if geom_engine.contains(neighbor_geom.constGet()):
                    layer.deleteFeature(neighbor_id)
                # check if neighbor contains feature
                if geom_engine.within(neighbor_geom.constGet()):
                    layer.deleteFeature(fid)


@cancel_on_entry
def fix_null_link_ids(layer, valid_link_ids):
    for feat in layer.getFeatures(
            QgsFeatureRequest().setFilterExpression("LinkId IS NULL")
    ):
        check_feedback()
        feat["LinkId"] = valid_link_ids.pop(0)
        layer.updateFeature(feat)
    update_progress("- Fixed null link ids")


@cancel_on_entry
def fix_nonsense_link_ids(layer, link_id_max, valid_link_ids):
    invalid_link_id_range_expr = f"LinkId > {link_id_max} or LinkId < 1"
    for feat in layer.getFeatures(
            QgsFeatureRequest().setFilterExpression(
                invalid_link_id_range_expr
            )
    ):
        check_feedback()
        feat["LinkId"] = valid_link_ids.pop(0)
        layer.updateFeature(feat)
    update_progress("- Fixed nonsense link ids")


@cancel_on_entry
def fix_duplicate_link_ids(layer, valid_link_ids, raw_ids):
    duplicate_ids = [
        item for item, count in Counter(raw_ids).items() if count > 1
    ]
    for dupe_id in duplicate_ids:
        check_feedback()
        equals_dupe_id_expr = f"LinkId = {dupe_id}"
        for feat in list(
                layer.getFeatures(
                    QgsFeatureRequest().setFilterExpression(
                        equals_dupe_id_expr
                    )
                )
        )[1:]:
            check_feedback()
            feat["LinkId"] = valid_link_ids.pop(0)
            layer.updateFeature(feat)
    update_progress("- Fixed duplicate link ids")


@cancel_on_entry
def fix_link_ids(layer):
    link_id_max = layer.featureCount() * 2
    raw_ids = list(
        flatten_collection(QgsVectorLayerUtils.getValues(layer, "LinkId"))
    )
    used_ids = set(raw_ids)
    valid_link_ids = list(set(range(1, link_id_max)) - used_ids)
    with edit(layer):
        # fix null link ids
        fix_null_link_ids(layer, valid_link_ids)
        # fix nonsense link ids
        fix_nonsense_link_ids(layer, link_id_max, valid_link_ids)
        # fix duplicate link ids
        fix_duplicate_link_ids(layer, valid_link_ids, raw_ids)


@cancel_on_entry
def fix_feature_names(layer, channel_layer):
    # 1) Build canonicalized name counts (trim + upper) to find duplicates robustly
    name_counts = Counter()
    for f in layer.getFeatures():
        check_feedback()

        n = f["Name"]
        if n is not None:
            s = str(n).strip()
            if s:
                name_counts[s.upper()] += 1

    duplicate_set = {n for n, c in name_counts.items() if c > 1}

    # 2) Single edit session: standardize, fix dupes, fill missing
    with edit(layer):
        for feature in layer.getFeatures():
            check_feedback()

            link_id = f"{int(feature['LinkId'])}"
            cur_name = feature["Name"]

            # Missing or empty name -> derive from channel_layer or fallback
            if cur_name is None or not str(cur_name).strip():
                sds_name, district_code = _infer_channel_name(feature, channel_layer)
                if sds_name and district_code:
                    new_name = f"{sds_name} {district_code} {link_id}"
                else:
                    new_name = f"FEATURE {link_id}"
            else:
                base = str(cur_name).strip().upper()
                if base in duplicate_set:
                    new_name = f"{base} {link_id}"
                else:
                    new_name = base

            if new_name is not None and new_name != cur_name:
                feature["Name"] = new_name
                layer.updateFeature(feature)

    update_progress("- Standardized feature names (trim/upper), de-duplicated with LinkId, filled null/empty")


@cancel_on_entry
def _infer_channel_name(feature, channel_layer):
    """
    Returns (SDSFEATURENAME_UPPER, usaceDistrictCode_UPPER) or (None, None)
    Uses bbox prefilter + precise intersects check.
    """
    g = feature.geometry()
    if g is None or g.isEmpty():
        return None, None

    request = QgsFeatureRequest().setFilterRect(g.boundingBox())
    for overlap in channel_layer.getFeatures(request):
        check_feedback()
        og = overlap.geometry()
        if og and not og.isEmpty() and g.intersects(og):
            names = overlap.fields().names()
            if "SDSFEATURENAME" in names and "usaceDistrictCode" in names:
                sds_raw = overlap["SDSFEATURENAME"]
                dist_raw = overlap["usaceDistrictCode"]
                sds = str(sds_raw).strip().upper() if sds_raw is not None else None
                dist = str(dist_raw).strip().upper() if dist_raw is not None else None
                return sds if sds else None, dist if dist else None
    return None, None


@cancel_on_entry
def set_ehydro_and_cpt_attributes(layer, ncf_layer):
    add_unique_field(layer, "ChannelRea", QVariant.String)
    idx = QgsSpatialIndex(ncf_layer.getFeatures(), flags=QgsSpatialIndex.FlagStoreFeatureGeometries)
    chan_idx = layer.fields().indexFromName("ChannelRea")
    depth_idx = layer.fields().indexFromName("DepthFt")

    def get_len_mi(geom):
        return DISTANCE_AREA.convertLengthMeasurement(
            geom.length(), QgsUnitTypes.DistanceMiles
        )

    with edit(layer):
        for line_feat in layer.getFeatures():
            check_feedback()

            line_geom = line_feat.geometry()
            if line_geom is None or line_geom.isEmpty():
                line_feat.setAttribute(chan_idx, None)
                layer.updateFeature(line_feat)
                continue

            line_len_mi = get_len_mi(line_geom)
            if line_len_mi <= 0.0:
                line_feat.setAttribute(chan_idx, None)
                layer.updateFeature(line_feat)
                continue

            best_feat = None
            best_ratio = 0.0

            # candidate polygons whose bbox intersects this line
            cand_ids = idx.intersects(line_geom.boundingBox())

            for fid in cand_ids:
                check_feedback()

                ncf_feat = ncf_layer.getFeature(fid)
                poly_geom = ncf_feat.geometry()
                if poly_geom is None or poly_geom.isEmpty():
                    continue

                inter = poly_geom.intersection(line_geom)
                if inter is None or inter.isEmpty():
                    continue

                overlap_len_mi = get_len_mi(inter)
                if overlap_len_mi <= 0.0:
                    continue

                overlap_ratio = overlap_len_mi / line_len_mi

                if overlap_ratio > best_ratio:
                    best_ratio = overlap_ratio
                    best_feat = ncf_feat

            # Decide once per line, after examining all candidates
            if best_feat is not None and best_ratio > 0.5:
                try:
                    line_feat.setAttribute(chan_idx, best_feat["channelreachidpk"])
                    if depth_idx != -1:
                        line_feat.setAttribute(
                            depth_idx,
                            (best_feat["depthmaintained"] or 99)
                        )
                except:
                    line_feat.setAttribute(chan_idx, best_feat["channelrea"])
                    if depth_idx != -1:
                        line_feat.setAttribute(
                            depth_idx,
                            (best_feat["depthmaint"] or 99)
                        )
            else:
                line_feat.setAttribute(chan_idx, None)

            layer.updateFeature(line_feat)

        update_progress("- Joined eHydro & CPT attributes by location (> 50% line overlap)")


@cancel_on_entry
def add_lat_lon_attributes(layer):
    lat_lon_fields = [
        QgsField("Latitude", QVariant.Double),
        QgsField("Longitude", QVariant.Double)
    ]
    layer.dataProvider().addAttributes(lat_lon_fields)
    layer.updateFields()
    with edit(layer):
        for feature in layer.getFeatures():
            check_feedback()
            point = QgsPointXY(QgsGeometry.asPoint(feature.geometry()))
            feature["Longitude"] = float('%.6f' % (point.x()))
            feature["Latitude"] = float('%.6f' % (point.y()))
            layer.updateFeature(feature)
    update_progress("- Lat-lon attributes created & assigned to nodes")


@cancel_on_entry
def assign_international_link_depths(layer):
    international_links_expr = "LinkType = 'International' or LinkType = 'Internat River'"
    for feat in layer.getFeatures(
            QgsFeatureRequest().setFilterExpression(
                international_links_expr
            )
    ):
        check_feedback()
        feat["DepthFt"] = 99
        layer.updateFeature(feat)


@cancel_on_entry
def assign_all_other_link_depths(layer):
    invalid_depth_values = [None, '', 0]
    for feature in layer.getFeatures():
        check_feedback()
        if feature["DepthFt"] in invalid_depth_values:
            neighbor_expr = (
                f"(i = '{feature['i']}' OR "
                f"j = '{feature['i']}' OR "
                f"i = '{feature['j']}' OR "
                f"j = '{feature['j']}') AND "
                f"DepthFt IS NOT NULL AND "
                f"DepthFt != '' AND "
                f"DepthFt != 0"
            )
            neighbors = layer.getFeatures(
                QgsFeatureRequest().setFilterExpression(neighbor_expr)
            )
            depths = [
                neighbor["DepthFt"] for neighbor in neighbors
                if isinstance(neighbor["DepthFt"], (int, float))
            ]
            if depths:
                feature["DepthFt"] = max(depths)
                layer.updateFeature(feature)


@cancel_on_entry
def assign_depths(layer):
    with edit(layer):
        # assign international link depths
        assign_international_link_depths(layer)
        # assign all other link depths
        assign_all_other_link_depths(layer)
    update_progress(f"- Link depths assigned")

@cancel_on_entry
def add_state_attributes(layer, state_input, state_abbr, state_fips):
    if (
            QgsWkbTypes.geometryType(
                state_input.wkbType()
            ) != QgsWkbTypes.PolygonGeometry
            or
            state_input.featureCount() == 0
    ):
        raise QgsProcessingException("State layer failed to load!")
    if state_abbr and state_fips:
        layer.dataProvider().addAttributes(
            [
                QgsField("StateAbbr", QVariant.String),
                QgsField("StateFIPS", QVariant.Int)
            ]
        )
        layer.updateFields()
        state_params = {
            'INPUT': layer,
            'PREDICATE': [0],
            'JOIN': state_input,
            'JOIN_FIELDS': [],
            'METHOD': 2,
            'DISCARD_NONMATCHING': False,
            'PREFIX': ''
        }
        state_layer = run_alg(
            "native:joinattributesbylocation",
            state_params, False
        )
        with edit(layer):
            for state_feat in state_layer.getFeatures():
                check_feedback()
                if (
                        state_feat[state_abbr] is not None
                        and
                        state_feat[state_fips] is not None
                ):
                    for feat in layer.getFeatures(
                            QgsFeatureRequest().setFilterRect(
                                state_feat.geometry().boundingBox()
                            )
                    ):
                        check_feedback()
                        feat.setAttribute("StateAbbr", state_feat[state_abbr])
                        feat.setAttribute("StateFIPS", state_feat[state_fips])
                        layer.updateFeature(feat)
        update_progress(
            "- State Abbreviation/FIPS attributes created & assigned to nodes"
        )
    else:
        update_progress()


@cancel_on_entry
def add_county_attributes(layer, county_input, county_name, county_fips):
    if (
            QgsWkbTypes.geometryType(
                county_input.wkbType()
            ) != QgsWkbTypes.PolygonGeometry
            or
            county_input.featureCount() == 0
    ):
        raise QgsProcessingException("County layer failed to load!")
    if county_name and county_fips:
        layer.dataProvider().addAttributes(
            [
                QgsField("CountyName", QVariant.String),
                QgsField("CountyFIPS", QVariant.Int)
            ]
        )
        layer.updateFields()
        county_params = {
            'INPUT': layer,
            'PREDICATE': [0],
            'JOIN': county_input,
            'JOIN_FIELDS': [],
            'METHOD': 2,
            'DISCARD_NONMATCHING': False,
            'PREFIX': '',
            'OUTPUT': 'TEMPORARY_OUTPUT'
        }
        county_layer = run_alg(
            "native:joinattributesbylocation",
            county_params, False
        )
        with edit(layer):
            for county_feat in county_layer.getFeatures():
                check_feedback()
                if (
                        county_feat[county_name] is not None
                        and
                        county_feat[county_fips] is not None
                ):
                    for feat in layer.getFeatures(
                            QgsFeatureRequest().setFilterRect(
                                county_feat.geometry().boundingBox()
                            )
                    ):
                        check_feedback()
                        feat.setAttribute(
                            "CountyName", county_feat[county_name]
                        )
                        feat.setAttribute(
                            "CountyFIPS", county_feat[county_fips]
                        )
                        layer.updateFeature(feat)
        update_progress(
            "- County Name attributes created & assigned to nodes"
        )
    else:
        update_progress()


@cancel_on_entry
def add_country_attributes(layer, country_input, country_abbr):
    if (
            QgsWkbTypes.geometryType(
                country_input.wkbType()
            ) != QgsWkbTypes.PolygonGeometry
            or
            country_input.featureCount() == 0):
        raise QgsProcessingException("Country layer failed to load!")
    if country_abbr:
        layer.dataProvider().addAttributes(
            [QgsField("CountryAbbr", QVariant.String)]
        )
        layer.updateFields()
        country_params = {
            'INPUT': layer,
            'PREDICATE': [0],
            'JOIN': country_input,
            'JOIN_FIELDS': [],
            'METHOD': 2,
            'DISCARD_NONMATCHING': False,
            'PREFIX': '',
            'OUTPUT': 'TEMPORARY_OUTPUT'
        }
        country_layer = run_alg(
            "native:joinattributesbylocation",
            country_params, False
        )
        with edit(layer):
            for country_feat in country_layer.getFeatures():
                check_feedback()
                if country_feat[country_abbr] is not None:
                    for feat in layer.getFeatures(
                            QgsFeatureRequest().setFilterRect(
                                country_feat.geometry().boundingBox()
                            )
                    ):
                        check_feedback()
                        feat.setAttribute(
                            "CountryAbbr", country_feat[country_abbr]
                        )
                        layer.updateFeature(feat)
        update_progress(
            "- Country Abbreviation/FIPS attributes created & assigned to nodes"
        )
    else:
        update_progress()


@cancel_on_entry
def get_end_points(line_geom):
    if line_geom.isEmpty():
        return []
    c = line_geom.constGet()
    return [c.startPoint(), c.endPoint()]


@cancel_on_entry
def points_equal(point_a, point_b):
    return point_a.x() == point_b.x() and point_a.y() == point_b.y()


@cancel_on_entry
def repair_closed_loop_lines(layer):
    def select_closed_loop_lines():
        out = []
        for f in layer.getFeatures():
            check_feedback()

            line_geom = f.geometry()
            end_points = get_end_points(line_geom)
            if not end_points:
                continue
            if points_equal(end_points[0], end_points[-1]):
                out.append(f)
        return out

    with edit(layer):
        closed_loop_lines = select_closed_loop_lines()
        for loop_line in closed_loop_lines:
            check_feedback()

            vertices = list(loop_line.geometry().vertices())
            if len(vertices) < 3:
                layer.deleteFeature(loop_line.id())
            else:
                new_geom = QgsGeometry(QgsLineString(vertices[:-1]))
                loop_line.setGeometry(new_geom)
                layer.updateFeature(loop_line)


@cancel_on_entry
def merge_short_lines(line_layer, min_length_miles, tol=1e-9):
    protected = set()

    def select_short_lines():
        out = []
        for f in line_layer.getFeatures():
            check_feedback()

            line_geom = f.geometry()
            end_points = get_end_points(line_geom)
            if not end_points or f.id() in protected:
                continue
            if get_geom_len_mi(line_geom) <= min_length_miles:
                out.append(f)
        return out

    def get_line_vertices(geom):
        return list(geom.vertices())

    def merge_at_shared_end_point(geom_a, geom_b):
        a = get_line_vertices(geom_a)
        b = get_line_vertices(geom_b)

        shared = None
        for p1 in (a[0], a[-1]):
            check_feedback()

            for p2 in (b[0], b[-1]):
                check_feedback()

                if points_equal(p1, p2):
                    shared = p1
                    break
            if shared is not None:
                break

        if shared is None:
            raise ValueError("Lines do not share an endpoint")

        def orient_line1(verts):
            if points_equal(verts[-1], shared):
                return verts  # already ends at shared
            if points_equal(verts[0], shared):
                return list(reversed(verts))  # reverse so it ends at shared
            raise ValueError("Shared point is not an endpoint of line 1")

        def orient_line2(verts):
            if points_equal(verts[0], shared):
                return verts  # already starts at shared
            if points_equal(verts[-1], shared):
                return list(reversed(verts))  # reverse so it starts at shared
            raise ValueError("Shared point is not an endpoint of line 2")

        v1_oriented = orient_line1(a)
        v2_oriented = orient_line2(b)

        merged_vertices = v1_oriented + v2_oriented[1:]
        return QgsGeometry(QgsLineString(merged_vertices))

    def update_and_delete(sl, nn, merge_geom):
        if merge_geom is None:
            protected.add(sl.id())
            return False

        nn.setGeometry(merge_geom)
        if not line_layer.updateFeature(nn):
            protected.add(sl.id())
            return False

        line_layer.deleteFeature(sl.id())
        return True

    def set_merged_geom(sl, members):
        nf_id = next(fid for fid in members if fid != sl.id())
        neighbor = line_layer.getFeature(nf_id)
        if not neighbor or not neighbor.isValid():
            protected.add(sl.id())
            return False
        merged = merge_at_shared_end_point(sl.geometry(), neighbor.geometry())
        return update_and_delete(sl, neighbor, merged)

    def build_node_map():
        node = {}  # (x,y) -> [fid,...]
        for feat in line_layer.getFeatures():
            check_feedback()

            fid = feat.id()
            end_points = get_end_points(feat.geometry())
            if not end_points:
                continue
            starting_xy = end_points[0].x(),end_points[0].y()
            ending_xy = end_points[-1].x(),end_points[-1].y()
            node.setdefault(starting_xy, []).append(fid)
            node.setdefault(ending_xy, []).append(fid)
        return node

    changed = True
    with edit(line_layer):
        while changed:
            check_feedback()

            changed = False

            node_map = build_node_map()
            short_lines = select_short_lines()
            if short_lines:
                for short_line in short_lines:
                    check_feedback()

                    if not short_line or not short_line.isValid():
                        continue

                    short_line_geom = short_line.geometry()
                    short_line_end_points = get_end_points(short_line_geom)

                    start_xy = short_line_end_points[0].x(),short_line_end_points[0].y()
                    end_xy = short_line_end_points[-1].x(),short_line_end_points[-1].y()

                    members_start = set(node_map.get(start_xy, []))
                    members_end = set(node_map.get(end_xy, []))
                    if not members_start and not members_end:
                        continue

                    if members_start and len(members_start) == 2:
                        changed = set_merged_geom(short_line, members_start)
                    elif members_end and len(members_end) == 2:
                        changed = set_merged_geom(short_line, members_end)
                    elif get_geom_len_mi(short_line_geom) > 0.0:
                        protected.add(short_line.id())
                        continue

                    if changed:
                        break

    update_progress("- Merged short lines")
    return protected


class QualityControlAlgorithm(QgsProcessingAlgorithm):
    def tr(self, string):
        # Returns a translatable string with the self.tr() function.
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        # Must return a new copy of your algorithm.
        return QualityControlAlgorithm()

    def name(self):
        # Returns the unique algorithm name.
        return 'qualitycontrol'

    def displayName(self):
        # Returns the translated algorithm name.
        return self.tr('Quality Control for Waterway Networks')

    def group(self):
        # Returns the name of the group this algorithm belongs to.
        return self.tr('Quality Control')

    def groupId(self):
        # Returns the unique ID of the group this algorithm belongs to.
        return 'qaqc'

    def shortHelpString(self):
        # Returns a localized short help string for the algorithm.
        desc = '''The "Quality Control for Waterway Networks" Processing Algorithm is an automated process to update the U.S. Army Corps of Engineer’s (USACE) Engineer Research and Development Center (ERDC) Waterway Network. After a user introduces desired changes to a line layer representing the waterways, the algorithm creates a fully connected network, and controls topology quality.  In addition, the algorithm updates waterway depths and geometries based on the most recent version of the USACE National Channel Framework (NCF), and performs spatial joins of network nodes with other various sources of data.

        <hr>
        <b>INPUTS</b>

        '''
        desc += '• The <b>Input Waterway layer</b> is your input. Accepts line vector layers. Usually will be your Waterway Network layer.<br>'
        desc += '• The <b>ChannelReach layer</b> is the layer containing NCF data. Accepts polygon vector layers. Usually is named "ChannelReach".<br>'
        desc += '• The <b>State layer</b> is an optional layer containing data of various states. Accepts polygon vector layers. State abbreviations/FIPS codes will not be added to the node layer if left blank.<br>'
        desc += '• The <b>County layer</b> is an optional layer containing data of various counties. Accepts polygon vector layers. County names will not be added to the node layer if left blank.<br>'
        desc += '• The <b>Country layer</b> is an optional layer containing data of various countries. Accepts polygon vector layers. Country abbreviations will not be added to the node layer if left blank.<br>'
        desc += '''• If the <b>Bypass the "Split Waterway Lines Based On ChannelReach" step?</b> is checked, the geometries of the input layer will not change to reflect NCF geometries. Waterway network depths will still be updated. By checking this box, the execution of the algorithm is faster.

        At the time of developing this tool, public versions of the above-mentioned inputs can be found in the following links:<br>'''
        desc += '• NCF ChannelReach: <a href="https://services7.arcgis.com/n1YM8pTrFmm7L4hs/ArcGIS/rest/services/National_Channel_Framework/FeatureServer/2">https://services7.arcgis.com/n1YM8pTrFmm7L4hs/ArcGIS/rest/services/National_Channel_Framework/FeatureServer/2</a><br>'
        desc += '• U.S. States: <a href="https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_5m.zip">https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_5m.zip</a><br>'
        desc += '''• U.S. Counties: <a href="https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_5m.zip">https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_5m.zip</a>

        The user may substitute the State, County, and Country layers with other polygon layers. The algorithm associates a waterway network node with a polygon's area. Fields to be joined are selected within the Advanced Parameters.

        <hr>
        <b>SETTINGS</b>

        • The <b>Minimum geometry length</b> is the minimum desired length of a link, in miles (LenMiles). All links with LenMiles below this threshold will be deleted. Recommended value: 0.02
        '''
        desc += 'In the Advanced Parameters,<br>'
        desc += '• The <b>"Split Waterway Lines Based On ChannelReach" length</b> is the minimum length of a link separation, in miles. If a newly-created link with length below this threshold would be made during the "Split Waterway Lines Based On ChannelReach" step, it is instead undone. Recommended value: 0.03<br>'
        desc += '• The <b>Snapping tolerance</b> controls how close line endpoints need to be before they are snapped. If the Disconnected Islands step detects islands, this will be iteratively multiplied until no islands are detected. Recommended value: 0.0002<br>'
        desc += '• The <b>Disconnected Islands tolerance</b> detects any groups of links in the input layer that are not connected to each other ("islands") within the threshold. Recommended value: 0.000001<br>'
        desc += '''• All parameters in the <b>FIELDS</b> section are optional. Accepts any field types. Columns involving their respective layer will not be added to the node layer if left blank.

        The USACE-ERDC waterway network is non-authoritative product created by the Coastal and Hydraulics Laboratory (CHL) for research purposes.'''
        return self.tr(desc)

    def initAlgorithm(self, config=None):
        # Here we define the inputs and outputs of the algorithm.
        # https://gis.stackexchange.com/questions/377793/how-to-group-parameters-in-pyqgis-processing-plugin
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                'INPUT',
                self.tr('<br><b>INPUTS</b><br><br>Input Waterway layer'),
                types=[QgsProcessing.TypeVectorLine]
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                'CHANNELREACH_INPUT', self.tr('ChannelReach layer'),
                types=[QgsProcessing.TypeVectorPolygon]
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                'STATE_INPUT', self.tr('State layer'),
                types=[QgsProcessing.TypeVectorPolygon], optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                'COUNTY_INPUT', self.tr('County layer'),
                types=[QgsProcessing.TypeVectorPolygon], optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                'COUNTRY_INPUT', self.tr('Country layer'),
                types=[QgsProcessing.TypeVectorPolygon], optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                'BYPASS_SPLIT',
                self.tr(
                    'Bypass the "Split Waterway Lines Based On ChannelReach" step?'
                )
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                'MIN_LENGTH',
                self.tr(
                    '<hr><br><b>SETTINGS</b><br><br>Minimum geometry length (mi)'
                ),
                type=1, defaultValue=0.02, minValue=0.0
            )
        )
        snap_tol = QgsProcessingParameterDistance(
            'SNAP_TOL', self.tr('Snapping tolerance'), defaultValue=0.0002,
            parentParameterName='INPUT', minValue=0.0
        )
        snap_tol.setFlags(
            snap_tol.flags() | QgsProcessingParameterDefinition.FlagAdvanced
        )
        self.addParameter(snap_tol)
        island_tol = QgsProcessingParameterDistance(
            'ISLAND_TOL', self.tr('Disconnected Islands tolerance'),
            defaultValue=0.000001, parentParameterName='INPUT', minValue=0.0
        )
        island_tol.setFlags(
            island_tol.flags() | QgsProcessingParameterDefinition.FlagAdvanced
        )
        self.addParameter(island_tol)
        state_abbr = QgsProcessingParameterField(
            'STATE_ABBR', '<hr><br><b>FIELDS</b><br><br>State Abbreviation',
            type=QgsProcessingParameterField.Any,
            parentLayerParameterName='STATE_INPUT', allowMultiple=False,
            optional=True
        )
        state_abbr.setFlags(
            state_abbr.flags() | QgsProcessingParameterDefinition.FlagAdvanced
        )
        self.addParameter(state_abbr)
        state_fips = QgsProcessingParameterField(
            'STATE_FIPS', 'State FIPS', type=QgsProcessingParameterField.Any,
            parentLayerParameterName='STATE_INPUT', allowMultiple=False,
            optional=True
        )
        state_fips.setFlags(
            state_fips.flags() | QgsProcessingParameterDefinition.FlagAdvanced
        )
        self.addParameter(state_fips)
        county_name = QgsProcessingParameterField(
            'COUNTY_NAME', 'County Name', type=QgsProcessingParameterField.Any,
            parentLayerParameterName='COUNTY_INPUT', allowMultiple=False,
            optional=True
        )
        county_name.setFlags(
            county_name.flags() | QgsProcessingParameterDefinition.FlagAdvanced
        )
        self.addParameter(county_name)
        county_fips = QgsProcessingParameterField(
            'COUNTY_FIPS', 'County FIPS', type=QgsProcessingParameterField.Any,
            parentLayerParameterName='COUNTY_INPUT', allowMultiple=False,
            optional=True
        )
        county_fips.setFlags(
            county_fips.flags() | QgsProcessingParameterDefinition.FlagAdvanced
        )
        self.addParameter(county_fips)
        country_abbr = QgsProcessingParameterField(
            'COUNTRY_ABBR', 'Country Abbreviation',
            type=QgsProcessingParameterField.Any,
            parentLayerParameterName='COUNTRY_INPUT', allowMultiple=False,
            optional=True
        )
        country_abbr.setFlags(
            country_abbr.flags() | QgsProcessingParameterDefinition.FlagAdvanced
        )
        self.addParameter(country_abbr)
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                'OUTPUT', self.tr(
                    '<hr><br><b>OUTPUTS</b><br><br>Links output'
                ),
                type=QgsProcessing.TypeVectorLine
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                'NODE_OUTPUT', self.tr('Nodes output'),
                type=QgsProcessing.TypeVectorPoint
            )
        )
        self.addOutput(
            QgsProcessingOutputString('FLAGGED', self.tr('Flagged LinkIDs'))
        )

    def processAlgorithm(self, parameters, context, feedback):
        # Here is where the processing itself takes place.
        global CONTEXT
        CONTEXT = context
        global FEEDBACK
        FEEDBACK = feedback
        global START_TIME
        START_TIME = time.time()
        try:
            input_layer = self.parameterAsVectorLayer(
                parameters, 'INPUT',CONTEXT
            )
            if input_layer is None or \
                QgsWkbTypes.geometryType(
                    input_layer.wkbType()
                ) != QgsWkbTypes.LineGeometry or \
                input_layer.featureCount() == 0:
                raise QgsProcessingException("Input layer failed to load")
            global DISTANCE_AREA
            DISTANCE_AREA = QgsDistanceArea()
            QgsProject.instance().setDistanceUnits(QgsUnitTypes.DistanceMiles)
            bypass_split = self.parameterAsBoolean(
                parameters, 'BYPASS_SPLIT', CONTEXT
            )
            min_geom_length = self.parameterAsDouble(
                parameters, 'MIN_LENGTH', CONTEXT
            )
            snapping_tolerance = self.parameterAsDouble(
                parameters, 'SNAP_TOL', CONTEXT
            )
            island_tol = self.parameterAsDouble(
                parameters, 'ISLAND_TOL', CONTEXT
            )
            flagged = []

            ###################################################################
            # STEP 1: Modify Network Links
            ###################################################################
            # 1a) Get latest NCF from web, copy it, then clean it
            cleaned_ncf_layer = clean_layer(
                copy_layer(
                    self.parameterAsVectorLayer(
                        parameters, 'CHANNELREACH_INPUT', CONTEXT
                    )
                )
            )
            update_progress("- Fetched latest NCF")
            # 1b) Copy the input layer then clean it
            cleaned_input_layer = clean_layer(copy_layer(input_layer))
            update_progress("- Cleaned input layer")
            cleaned_input_layer = run_alg(
                "native:multiparttosingleparts", {"INPUT": cleaned_input_layer}
            )
            # 1c) Fix overlaps before line splitting
            fix_overlaps(cleaned_input_layer)
            update_progress("- Fixed overlapping geometries")
            # 1d) Split input layer with NCF layer if no bypass
            cleaned_input_layer = split_input_with_ncf_lines(
                bypass_split, cleaned_input_layer,
                copy_layer(cleaned_ncf_layer)
            )
            # 1e) Fix lines where start point and end point are the same XY point
            repair_closed_loop_lines(cleaned_input_layer)
            ###################################################################
            # STEP 2: QAQC (attributes)
            ###################################################################
            # 2a) Recalculate feature lengths in miles
            recalculate_feature_lengths(cleaned_input_layer)
            # 2b) Remove features where LenMiles is 0 or None
            zero_or_null_length_expr = f"LenMiles = {0} OR LenMiles IS NULL"
            remove_features_by_expression(
                cleaned_input_layer, zero_or_null_length_expr
            )
            update_progress("- Removed features where LenMiles is 0 or NULL")
            # 2c) Create spatial index
            run_alg(
                "native:createspatialindex",
                {'INPUT': cleaned_input_layer},
                False
            )
            update_progress("- Created spatial index")
            # 2d) Fix any LinkType typos
            fix_link_types(cleaned_input_layer, flagged)
            # 2e) Fix feature link ids
            fix_link_ids(cleaned_input_layer)
            # 2f) Fix feature names
            fix_feature_names(cleaned_input_layer, cleaned_ncf_layer)
            ###################################################################
            # STEP 3: QAQC (topology check)
            ###################################################################
            # 3a) Snap geometries (closest point)
            cleaned_input_layer = snap_geometries_with_snapper(
                cleaned_input_layer, snapping_tolerance
            )
            # Recalculate feature lengths after snap
            recalculate_feature_lengths(cleaned_input_layer)
            # 3b) Merge short lines
            protected_ids = merge_short_lines(
                cleaned_input_layer,
                min_geom_length
            )
            # Recalculate feature lengths after merge
            recalculate_feature_lengths(cleaned_input_layer)
            # 3c) Remove empty and very short geometries
            remove_empty_and_short_geometries(
                cleaned_input_layer, min_geom_length, protected_ids=protected_ids
            )
            # convert to single parts for snapping and reconnection of islands
            cleaned_input_layer = run_alg(
                "native:multiparttosingleparts", {"INPUT": cleaned_input_layer}
            )
            # 3d) Snap geometries after removing empty/short geometries (endpoint-to-endpoint only)
            cleaned_input_layer = clean_layer(cleaned_input_layer)
            cleaned_input_layer = snap_geometries(
                cleaned_input_layer, snapping_tolerance
            )
            # 3e) Reconnect islands
            cleaned_input_layer = reconnect_islands(
                cleaned_input_layer, island_tol, snapping_tolerance
            )
            ###################################################################
            # STEP 4: Joins with eHydro & CPT
            ###################################################################
            # Set eHydro & CPT attributes; join on 1-1 relation based on max overlap
            set_ehydro_and_cpt_attributes(
                cleaned_input_layer, cleaned_ncf_layer
            )
            # no longer needed at this point; free up memory
            delete_layer(cleaned_ncf_layer)
            ###################################################################
            # STEP 5: Create network
            ###################################################################
            # 5a) Use QGIS Networks plugin (Build Graph)
            nodes_params = {
                'RESEAU': copy_layer(cleaned_input_layer),
                'SENS': '',
                'IDENT': 0,
                'PREFIXE': '',
                'DECIMALES': 9
            }
            # no longer needed at this point; free up memory
            delete_layer(cleaned_input_layer)
            lines_layer, nodes_layer = self.Networks().build_graph(nodes_params)
            # 5b) Nodes: Add lat-lon attributes
            add_lat_lon_attributes(nodes_layer)
            # 5c) Create spatial index
            run_alg(
                "native:createspatialindex",
                {'INPUT': lines_layer},
                False
            )
            update_progress("- Created spatial index")
            ###################################################################
            # STEP 6: Assign link depths
            ###################################################################
            # Assign link depths
            assign_depths(lines_layer)
            ###################################################################
            # STEP 7: Set state, county, and country attributes
            ###################################################################
            # 7a) Set state attributes
            state_input = self.parameterAsVectorLayer(
                parameters, 'STATE_INPUT', CONTEXT
            )
            if state_input is not None:
                state_abbr = self.parameterAsString(
                    parameters, 'STATE_ABBR', CONTEXT
                )
                state_fips = self.parameterAsString(
                    parameters, 'STATE_FIPS', CONTEXT
                )
                add_state_attributes(
                    nodes_layer, state_input, state_abbr, state_fips
                )
            else:
                update_progress()
            # 7b) Set county attributes
            county_input = self.parameterAsVectorLayer(
                parameters, 'COUNTY_INPUT', CONTEXT
            )
            if county_input is not None:
                county_name = self.parameterAsString(
                    parameters, 'COUNTY_NAME', CONTEXT
                )
                county_fips = self.parameterAsString(
                    parameters, 'COUNTY_FIPS', CONTEXT
                )
                add_county_attributes(
                    nodes_layer, county_input, county_name, county_fips
                )
            else:
                update_progress()
            # 7c) Set country attributes
            country_input = self.parameterAsVectorLayer(
                parameters, 'COUNTRY_INPUT', CONTEXT
            )
            if country_input is not None:
                country_abbr = self.parameterAsString(
                    parameters, 'COUNTRY_ABBR', CONTEXT
                )
                add_country_attributes(
                    nodes_layer, country_input, country_abbr
                )
            else:
                update_progress()
            ###################################################################
            # Cleanup and finalization of results
            ###################################################################
            # Remove Plot, Domestic, Deepdraft, networkGrp, and fid attributes
            remove_feature_attribute_by_name(lines_layer, "Plot")
            remove_feature_attribute_by_name(lines_layer, "Domestic")
            remove_feature_attribute_by_name(lines_layer, "Deepdraft")
            remove_feature_attribute_by_name(lines_layer, "networkGrp")
            remove_feature_attribute_by_name(lines_layer, "fid")
            remove_feature_attribute_by_name(nodes_layer, "fid")
            update_progress()
            # Open attributes table, abacus icon, update existing attribute, LenMiles, Geometry, re-calculate $length
            recalculate_feature_lengths(lines_layer)
            update_progress()
            # Refactor fields
            refactor_params = {
                'INPUT': lines_layer,
                'FIELDS_MAPPING': [
                    {'alias': '', 'comment': '', 'expression': '"LinkId"',
                     'length': 0, 'name': 'LinkId',
                     'precision': 0,
                     'sub_type': 0, 'type': 4, 'type_name': 'int8'},
                    {'alias': '', 'comment': '', 'expression': '"Name"',
                     'length': 254, 'name': 'Name', 'precision': 0,
                     'sub_type': 0, 'type': 10, 'type_name': 'text'},
                    {'alias': '', 'comment': '', 'expression': '"LenMiles"',
                     'length': 20, 'name': 'LenMiles',
                     'precision': 4, 'sub_type': 0, 'type': 6,
                     'type_name': 'double precision'},
                    {'alias': '', 'comment': '', 'expression': '"DepthFt"',
                     'length': 20, 'name': 'DepthFt',
                     'precision': 4,
                     'sub_type': 0, 'type': 6,
                     'type_name': 'double precision'},
                    {'alias': '', 'comment': '', 'expression': '"LinkType"',
                     'length': 80, 'name': 'LinkType',
                     'precision': 0, 'sub_type': 0, 'type': 10,
                     'type_name': 'text'},
                    {'alias': '', 'comment': '', 'expression': '"i"',
                     'length': 80, 'name': 'i', 'precision': 0,
                     'sub_type': 0, 'type': 10, 'type_name': 'text'},
                    {'alias': '', 'comment': '', 'expression': '"j"',
                     'length': 80, 'name': 'j', 'precision': 0,
                     'sub_type': 0, 'type': 10, 'type_name': 'text'},
                    {'alias': '', 'comment': '', 'expression': '"ij"',
                     'length': 80, 'name': 'ij', 'precision': 0,
                     'sub_type': 0, 'type': 10, 'type_name': 'text'},
                    {'alias': '', 'comment': '', 'expression': '"ChannelRea"',
                     'length': 25, 'name': 'ChannelRea',
                     'precision': 0, 'sub_type': 0, 'type': 10,
                     'type_name': 'text'}
                ]
            }
            lines_layer = run_alg("native:refactorfields", refactor_params)
            update_progress()
            # https://gis.stackexchange.com/questions/415841/changing-output-layers-name-in-qgis-processing-plugin
            date_str = date.today().strftime('%Y%m%d')
            # finalize link output
            link_sink, link_dest_id = self.parameterAsSink(
                parameters, 'OUTPUT', CONTEXT, lines_layer.fields(),
                lines_layer.wkbType(), lines_layer.sourceCrs()
            )
            link_layer_details = CONTEXT.layerToLoadOnCompletionDetails(
                link_dest_id
            )
            link_layer_details.name = f"{date_str}_network_links"
            for feat in lines_layer.getFeatures():
                check_feedback()
                link_sink.addFeature(feat, QgsFeatureSink.FastInsert)
            update_progress()
            # finalize node output
            node_sink, node_dest_id = self.parameterAsSink(
                parameters, 'NODE_OUTPUT', CONTEXT, nodes_layer.fields(),
                nodes_layer.wkbType(), nodes_layer.sourceCrs()
            )
            node_layer_details = CONTEXT.layerToLoadOnCompletionDetails(
                node_dest_id
            )
            node_layer_details.name = f"{date_str}_network_nodes"
            for feat in nodes_layer.getFeatures():
                check_feedback()
                node_sink.addFeature(feat, QgsFeatureSink.FastInsert)
            update_progress()
            # output details for flagged links
            if flagged:
                FEEDBACK.setProgressText(
                    "The following Links should be reviewed:")
                for x in flagged:
                    FEEDBACK.setProgressText(f"{x[0]}: {x[1]}")
            # return results
            return {
                'OUTPUT': link_dest_id,
                'NODE_OUTPUT': node_dest_id,
                'FLAGGED': ", ".join(
                    str(x[0]) + ": " + str(x[1]) for x in flagged)
            }
        except QgsProcessingCanceledException:
            FEEDBACK.pushInfo("User canceled process")
            return {}
        except Exception as e:
            FEEDBACK.pushInfo("Processing failed")
            FEEDBACK.pushInfo(f"{str(e)}")
            FEEDBACK.pushInfo(traceback.format_exc())
            return {}

    class Networks(QgsProcessingAlgorithm):
        @cancel_on_entry
        def calculate_libx(self, xtr, na, dec):
            xtr1 = xtr.transform(na)[0]
            xtr2 = xtr.transform(na)[1]
            string1 = str(
                int(xtr1 * 10 ** (dec - 3) + 180 * 10 ** (dec - 3))
            ).zfill(dec)
            string2 = str(
                int(xtr2 * 10 ** (dec - 3) + 180 * 10 ** (dec - 3))
            ).zfill(dec)
            return string1 + string2

        @cancel_on_entry
        def build_graph(self, parameters):
            reseau = parameters["RESEAU"]
            sens = parameters["SENS"]
            ident = parameters["IDENT"]
            prefixe = parameters["PREFIXE"]
            dec = parameters["DECIMALES"]

            layer = reseau
            nom_champs = []
            for i in layer.fields():
                check_feedback()

                nom_champs.append(i.name())
            if "i" not in nom_champs:
                layer.dataProvider().addAttributes(
                    [QgsField("i", QVariant.String)]
                )
            if "j" not in nom_champs:
                layer.dataProvider().addAttributes(
                    [QgsField("j", QVariant.String)]
                )
            if "ij" not in nom_champs:
                layer.dataProvider().addAttributes(
                    [QgsField("ij", QVariant.String)]
                )
            layer.updateFields()

            ida = layer.fields().indexFromName("i")
            idb = layer.fields().indexFromName("j")
            idij = layer.fields().indexFromName("ij")
            noeuds = {}
            src = QgsCoordinateReferenceSystem(layer.crs())
            dest = QgsCoordinateReferenceSystem("EPSG:4326")
            xtr = QgsCoordinateTransform(src, dest, QgsProject.instance())

            with edit(layer):
                for ligne in layer.getFeatures():
                    check_feedback()

                    if len(sens) == 0:
                        test_sens = '1'
                    else:
                        if ligne[sens] == '1':
                            test_sens = '1'
                        else:
                            test_sens = '0'
                    gligne = ligne.geometry()
                    if gligne.isEmpty():
                        layer.deleteFeature(ligne.id())
                        continue
                    if test_sens == '1':
                        if gligne.wkbType() in [
                            QgsWkbTypes.MultiLineString,
                            QgsWkbTypes.MultiLineStringZ
                        ]:
                            g = gligne.asMultiPolyline()
                            na = g[0][0]
                            liba = self.calculate_libx(xtr, na, dec)
                            if na.compare(g[-1][-1]):
                                nb = g[-1][-2]
                            else:
                                nb = g[-1][-1]
                            libb = self.calculate_libx(xtr, nb, dec)
                        elif gligne.wkbType() in [
                            QgsWkbTypes.LineString, QgsWkbTypes.LineStringZ
                        ]:
                            g = gligne.asPolyline()
                            na = g[0]
                            liba = self.calculate_libx(xtr, na, dec)
                            if na.compare(g[-1]):
                                nb = g[-2]
                            else:
                                nb = g[-1]
                            libb = self.calculate_libx(xtr, nb, dec)
                        else:
                            continue
                        if na not in noeuds:
                            noeuds[na] = (prefixe + liba, 1)
                        else:
                            noeuds[na] = (prefixe + liba, noeuds[na][1] + 1)
                        if nb not in noeuds:
                            noeuds[nb] = (prefixe + libb, 1)
                        else:
                            noeuds[nb] = (prefixe + libb, noeuds[nb][1] + 1)

            node_layer = QgsVectorLayer("Point", "temporary_points", "memory")
            node_provider = node_layer.dataProvider()
            node_provider.addAttributes(
                [
                    QgsField("num", QVariant.String),
                    QgsField("nb", QVariant.Int)
                ]
            )
            node_layer.updateFields()

            with edit(node_layer):
                for i, n in enumerate(noeuds):
                    check_feedback()

                    node = QgsFeature()
                    node.setGeometry(
                        QgsGeometry.fromPointXY(QgsPointXY(n[0], n[1]))
                    )
                    if ident == 0:
                        noeuds[n] = (prefixe + str(i), noeuds[n][1])
                    node.setAttributes([noeuds[n][0], noeuds[n][1]])
                    node_provider.addFeatures([node])

            with edit(layer):
                for i, ligne in enumerate(layer.getFeatures()):
                    check_feedback()

                    if len(sens) == 0:
                        test_sens = '1'
                    else:
                        if ligne[sens] == '1':
                            test_sens = '1'
                        else:
                            test_sens = '0'
                    if test_sens == '1':
                        gligne = ligne.geometry()
                        if gligne.wkbType() in [
                            QgsWkbTypes.MultiLineString,
                            QgsWkbTypes.MultiLineStringZ]:
                            g = gligne.asMultiPolyline()
                            na = g[0][0]
                            if na.compare(g[-1][-1]):
                                nb = g[-1][-2]
                            else:
                                nb = g[-1][-1]
                        elif gligne.wkbType() in [
                            QgsWkbTypes.LineString, QgsWkbTypes.LineStringZ
                        ]:
                            g = gligne.asPolyline()
                            na = g[0]
                            if na.compare(g[-1]):
                                nb = g[-2]
                            else:
                                nb = g[-1]
                        else:
                            continue
                        if noeuds[na][0] != noeuds[nb][0]:
                            id = ligne.id()
                            valid = {
                                ida: str(noeuds[na][0]),
                                idb: str(noeuds[nb][0]),
                                idij: str(
                                    noeuds[na][0] + "-" + noeuds[nb][0]
                                )
                            }
                            layer.changeAttributeValues(id, valid)
            update_progress("- Built Graph with Networks plugin")
            return layer, node_layer