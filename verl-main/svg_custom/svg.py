import random
from dataclasses import dataclass, field
from defusedxml import ElementTree as etree


@dataclass(frozen=True)
class SVGConstraints:
    """Defines constraints for validating SVG documents.

    Attributes
    ----------
    max_svg_size : int, default=10000
        Maximum allowed size of an SVG file in bytes.
    allowed_elements : dict[str, set[str]]
        Mapping of the allowed elements to the allowed attributes of each element.
    """

    max_svg_size: int = 10000
    allowed_elements: dict[str, set[str]] = field(
        default_factory=lambda: {
            'common': {
                'id',
                'clip-path',
                'clip-rule',
                'color',
                'color-interpolation',
                'color-interpolation-filters',
                'color-rendering',
                'display',
                'fill',
                'fill-opacity',
                'fill-rule',
                'filter',
                'flood-color',
                'flood-opacity',
                'lighting-color',
                'marker-end',
                'marker-mid',
                'marker-start',
                'mask',
                'opacity',
                'paint-order',
                'stop-color',
                'stop-opacity',
                'stroke',
                'stroke-dasharray',
                'stroke-dashoffset',
                'stroke-linecap',
                'stroke-linejoin',
                'stroke-miterlimit',
                'stroke-opacity',
                'stroke-width',
                'transform',
            },
            'svg': {
                'width',
                'height',
                'viewBox',
                'preserveAspectRatio',
            },
            'g': {'viewBox'},
            'defs': set(),
            'symbol': {'viewBox', 'x', 'y', 'width', 'height'},
            'use': {'x', 'y', 'width', 'height', 'href'},
            'marker': {
                'viewBox',
                'preserveAspectRatio',
                'refX',
                'refY',
                'markerUnits',
                'markerWidth',
                'markerHeight',
                'orient',
            },
            'pattern': {
                'viewBox',
                'preserveAspectRatio',
                'x',
                'y',
                'width',
                'height',
                'patternUnits',
                'patternContentUnits',
                'patternTransform',
                'href',
            },
            'linearGradient': {
                'x1',
                'x2',
                'y1',
                'y2',
                'gradientUnits',
                'gradientTransform',
                'spreadMethod',
                'href',
            },
            'radialGradient': {
                'cx',
                'cy',
                'r',
                'fx',
                'fy',
                'fr',
                'gradientUnits',
                'gradientTransform',
                'spreadMethod',
                'href',
            },
            'stop': {'offset'},
            'filter': {
                'x',
                'y',
                'width',
                'height',
                'filterUnits',
                'primitiveUnits',
            },
            'feBlend': {'result', 'in', 'in2', 'mode'},
            'feFlood': {'result'},
            'feOffset': {'result', 'in', 'dx', 'dy'},
            'path': {'d'},
            'rect': {'x', 'y', 'width', 'height', 'rx', 'ry'},
            'circle': {'cx', 'cy', 'r'},
            'ellipse': {'cx', 'cy', 'rx', 'ry'},
            'line': {'x1', 'y1', 'x2', 'y2'},
            'polyline': {'points'},
            'polygon': {'points'},
        }
    )

    def validate_svg(self, svg_code: str) -> None:
        """Validates an SVG string against a set of predefined constraints.

        Parameters
        ----------
        svg_code : str
            The SVG string to validate.

        Raises
        ------
        ValueError
            If the SVG violates any of the defined constraints.
        """
        # Check file size
        if len(svg_code.encode('utf-8')) > self.max_svg_size:
            return False

        # Parse XML
        tree = etree.fromstring(
            svg_code.encode('utf-8'),
            forbid_dtd=True,
            forbid_entities=True,
            forbid_external=True,
        )

        elements = set(self.allowed_elements.keys())

        # Check elements and attributes
        for element in tree.iter():
            # Check for disallowed elements
            tag_name = element.tag.split('}')[-1]
            if tag_name not in elements:
                return False

            # Check attributes
            for attr, attr_value in element.attrib.items():
                # Check for disallowed attributes
                attr_name = attr.split('}')[-1]
                if (
                    attr_name not in self.allowed_elements[tag_name]
                    and attr_name not in self.allowed_elements['common']
                ):
                    return False

                # Check for embedded data
                if 'data:' in attr_value.lower():
                    return False
                if ';base64' in attr_value:
                    return False

                # Check that href attributes are internal references
                if attr_name == 'href':
                    if not attr_value.startswith('#'):
                        return False
        return True

def compute_score(data_source, solution_str, ground_truth) -> float:
    if not isinstance(solution_str, str) or not SVGConstraints().validate_svg(solution_str):
        return 0.0
    return get_score(solution_str, ground_truth)

def get_score(solution_str, ground_truth) -> float:
    return random.randint(0, 1)