
import { useSearchParams } from 'react-router-dom';
import FamilySignupPage from './FamilySignupPage';
import ElderSignupPage from '../../../../elder-app/src/pages/auth/ElderSignupPage';

const SignupPage = () => {
    const [searchParams] = useSearchParams();
    const role = searchParams.get('role');

    if (role === 'elder') {
        return <ElderSignupPage />;
    }

    return <FamilySignupPage />;
};

export default SignupPage;
